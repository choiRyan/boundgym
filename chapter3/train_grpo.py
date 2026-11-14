"""Chapter 3: GRPO on boundgym with TRL, single GPU (vLLM colocated, LoRA).

  .vllm-venv/bin/python chapter3/train_grpo.py --run smoke --max-steps 3
  ... same args ... --resume latest      # continue an interrupted run from its newest checkpoint

Training levels use seeds 0.. (frozen eval sets use >= 1,000,000). The reward is
boundgym.verify.reward through trl_hooks.boundgym_reward: 1.0 only for a move
plan that reaches the goal alive within max_moves.
"""
import argparse, json, os, sys, time
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True, help="run name → chapter3/runs/<run>")
ap.add_argument("--model", default="Qwen/Qwen3-4B")
ap.add_argument("--difficulties", default="2,3,4")
ap.add_argument("--n-per", type=int, default=2000, help="training levels per difficulty")
ap.add_argument("--max-steps", type=int, default=200)
ap.add_argument("--max-completion", type=int, default=8192)
ap.add_argument("--num-generations", type=int, default=8)
ap.add_argument("--prompts-per-step", type=int, default=2)
ap.add_argument("--lr", type=float, default=1e-5)
ap.add_argument("--lora-r", type=int, default=32)
ap.add_argument("--vllm-mem", type=float, default=0.45)
ap.add_argument("--shaped", action="store_true", help="partial credit (0.2 x progress) for failures")
ap.add_argument("--save-steps", type=int, default=25)
ap.add_argument("--no-mask-truncated", action="store_true",
                help="keep cut-off completions in the loss (reward 0) instead of masking them out")
ap.add_argument("--profile", action="store_true",
                help="write TRL's phase timings (vLLM.generate, sync_weights, ...) to runs/<run>/profile.jsonl")
ap.add_argument("--resume", default=None, metavar="CKPT",
                help="continue the run from runs/<run>/checkpoint-N (or 'latest'); training args must match args.json")
ap.add_argument("--model-only-ckpt", action="store_true",
                help="save adapter only at checkpoints (smaller, but resuming loses optimizer/RNG state)")
a = ap.parse_args()

from datasets import Dataset
from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer
from boundgym.trl_hooks import boundgym_reward, boundgym_reward_shaped, make_records

out = os.path.join(ROOT, "chapter3", "runs", a.run)
os.makedirs(out, exist_ok=True)
resume = None
if a.resume:
    if a.resume == "latest":
        ckpts = [d for d in os.listdir(out) if d.startswith("checkpoint-")]
        if not ckpts:
            sys.exit(f"no checkpoints in {out}")
        a.resume = max(ckpts, key=lambda d: int(d.split("-")[1]))
    resume = a.resume if os.path.isabs(a.resume) else os.path.join(out, os.path.basename(a.resume))
    if not os.path.isfile(os.path.join(resume, "trainer_state.json")):
        sys.exit(f"not a checkpoint: {resume}")
    # The data order and step accounting depend on these; refuse to resume with different ones.
    saved = json.load(open(os.path.join(out, "args.json")))
    same = ["model", "difficulties", "n_per", "max_completion", "num_generations", "prompts_per_step",
            "lr", "lora_r", "shaped", "no_mask_truncated"]
    diff = {k: (saved.get(k), getattr(a, k)) for k in same if saved.get(k) != getattr(a, k)}
    if diff:
        sys.exit(f"args differ from {out}/args.json (saved, now): {diff}")
    with open(os.path.join(out, "resumes.jsonl"), "a") as f:
        f.write(json.dumps({"wall": time.time(), "from": resume, "args": vars(a)}) + "\n")
else:
    json.dump(vars(a), open(os.path.join(out, "args.json"), "w"), indent=2)

if a.profile:
    # TRL times its phases but only reports them to wandb/mlflow/trackio; capture them locally instead.
    from trl.extras import profiling as _prof
    _pf = open(os.path.join(out, "profile.jsonl"), "a")

    def _log_metrics(self, duration):
        if self.is_main_process:
            _pf.write(json.dumps({"step": self.step, "name": self.name, "sec": duration, "wall": time.time()}) + "\n")
            _pf.flush()
    _prof.ProfilingContext._log_metrics = _log_metrics

ds = Dataset.from_list(make_records([int(d) for d in a.difficulties.split(",")], a.n_per)).shuffle(seed=0)

cfg = GRPOConfig(
    output_dir=out,
    max_steps=a.max_steps,
    learning_rate=a.lr,
    lr_scheduler_type="constant_with_warmup",
    warmup_steps=5,
    bf16=True,
    model_init_kwargs={"dtype": "bfloat16"},  # default loads fp32 (16 GB for 4B), leaving no room for vLLM
    gradient_checkpointing=True,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=a.prompts_per_step * a.num_generations,
    num_generations=a.num_generations,
    max_completion_length=a.max_completion,
    temperature=0.6, top_p=0.95, top_k=20,  # Qwen3 defaults, same as vLLM eval (generation_config)
    loss_type="dapo",
    beta=0.0,
    mask_truncated_completions=not a.no_mask_truncated,
    use_vllm=True,
    vllm_mode="colocate",
    vllm_gpu_memory_utilization=a.vllm_mem,
    vllm_enable_sleep_mode=True,
    vllm_max_model_length=a.max_completion + 1024,
    logging_steps=1,
    log_completions=True,
    num_completions_to_print=1,
    save_steps=a.save_steps,
    save_only_model=a.model_only_ckpt,  # full = + optimizer/scheduler/RNG (~0.5 GB for r=32 LoRA)
    report_to="none",
    seed=0,
)
peft_cfg = LoraConfig(r=a.lora_r, lora_alpha=2 * a.lora_r, lora_dropout=0.0, task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])

from transformers import TrainerCallback


class MetricsLog(TrainerCallback):
    """One JSON line per logged step → runs/<run>/metrics.jsonl (read by scripts/dashboard.py)."""
    def on_log(self, args, state, control, logs=None, **kw):
        if state.is_world_process_zero and logs:
            with open(os.path.join(out, "metrics.jsonl"), "a") as f:
                f.write(json.dumps({"step": state.global_step, "max_steps": state.max_steps,
                                    "wall": time.time(), **logs}) + "\n")


trainer = GRPOTrainer(
    model=a.model,
    reward_funcs=[boundgym_reward_shaped if a.shaped else boundgym_reward],
    args=cfg,
    train_dataset=ds,
    peft_config=peft_cfg,
    callbacks=[MetricsLog()],
)
t0 = time.time()
trainer.train(resume_from_checkpoint=resume)
trainer.save_model(os.path.join(out, "final"))
with open(os.path.join(out, "log_history.json"), "w") as f:
    json.dump(trainer.state.log_history, f, indent=1)
print(f"done in {(time.time() - t0) / 60:.1f} min → {out}")
