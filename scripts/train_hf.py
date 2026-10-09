"""HF Jobs compatible tiny-train: reads /data, writes /data/outputs + Hub.

High-context + multimodal aware (mirrors scripts/train.py):
  python scripts/train_hf.py --data data/processed/train.jsonl \\
      --out /content/outputs/tiny-50M --epochs 1 --config configs/tiny-50M.yaml
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def resolve_fsdp_optim(fsdp, optim):
    """FSDP needs sharded-optimizer-compatible setup: 8-bit adam is out."""
    fsdp = (fsdp or "").strip()
    if fsdp and optim == "adamw_8bit":
        print("FSDP + adamw_8bit incompatible -> falling back to adamw_torch "
              "(states are sharded across GPUs instead)", flush=True)
        optim = "adamw_torch"
    return fsdp, optim


def _apply_rope_scaling(model, rope_scaling, seq_len: int):
    """Write YaRN scaling into model.config BEFORE the final save.

    Training itself stays on plain RoPE; only the saved config.json carries
    the scaling, so HF inference and GGUF conversion extrapolate to
    effective_ctx = original_max_position_embeddings * factor (>=100000).
    """
    if not rope_scaling:
        return
    rs = dict(rope_scaling)
    if "rope_type" not in rs and "type" in rs:
        rs["rope_type"] = rs["type"]
    original = int(rs.get("original_max_position_embeddings", seq_len))
    if original < seq_len:
        rs["original_max_position_embeddings"] = original = seq_len
    model.config.rope_scaling = rs
    eff = int(original * float(rs.get("factor", 1.0)))
    print(f"rope_scaling saved to config.json: {rs} "
          f"-> effective context {eff} tokens", flush=True)


def build_model(model_cfg, resume_from=""):
    """Fresh random init, or continue from a checkpoint dir (self-learn cycles).

    Config still comes from --config (arch must match the checkpoint).
    """
    import torch
    from transformers import LlamaForCausalLM

    if resume_from:
        p = Path(resume_from)
        if not (p / "config.json").exists():
            raise SystemExit(f"--resume-from has no config.json: {resume_from}")
        print(f"resuming from checkpoint: {resume_from}", flush=True)
        model = LlamaForCausalLM.from_pretrained(
            str(p), torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32)
    else:
        model = LlamaForCausalLM(model_cfg)
    return model


def estimate_need_gb(num_params: int, fp16: bool = True, full_ckpt: bool = True) -> float:
    """Disk for checkpoints + margin.

    full_ckpt=True (default): weights + fp32 Adam states (~9GB alone at 1B),
    x2 transient rotation + margin.
    full_ckpt=False (save_only_model): weights only -> ~5x smaller need.
    """
    w = 2 if fp16 else 4
    per_ckpt = (w + 8) if full_ckpt else w
    return num_params * per_ckpt * 2 * 1.3 / 1e9


def _cache_dirs():
    import os

    return {
        "HF_HOME": os.getenv("HF_HOME", os.path.expanduser("~/.cache/huggingface")),
        "TMPDIR": os.getenv("TMPDIR", "/tmp"),
    }


def _dir_size(path, depth=0, _seen=None):
    total = 0
    try:
        with os.scandir(path) as it:
            for e in it:
                try:
                    if e.is_symlink():
                        continue
                    if e.is_file(follow_symlinks=False):
                        total += e.stat(follow_symlinks=False).st_size
                    elif e.is_dir(follow_symlinks=False) and depth < 2:
                        total += _dir_size(e.path, depth + 1)
                except OSError:
                    continue
    except OSError:
        pass
    return total


def print_disk_hogs():
    """On preflight failure: show WHERE the GBs actually are (panel lies)."""
    import os

    cands = ["/kaggle/working", os.path.expanduser("~/.cache"), "/tmp", "."]
    rows = []
    for c in cands:
        if os.path.exists(c):
            rows.append((c, _dir_size(c)))
    rows.sort(key=lambda r: -r[1])
    print("disk hogs (top-level):", flush=True)
    for c, s in rows[:8]:
        print(f"  {s / 1e9:.1f} GB  {c}", flush=True)


def check_disk_gb(path, need_gb: float):
    """Fail fast BEFORE training: check out dir AND cache/tmp mounts.

    (Last time the out dir had space but the HF-cache mount was full,
    killing a 61% run at checkpoint save.)
    """
    import shutil

    Path(path).mkdir(parents=True, exist_ok=True)
    targets = {"outputs": str(path)}
    targets.update(_cache_dirs())
    worst = None
    for label, p in targets.items():
        try:
            Path(p).mkdir(parents=True, exist_ok=True)
            free_gb = shutil.disk_usage(str(p)).free / 1e9
        except OSError:
            free_gb = 0.0
        print(f"disk: {free_gb:.1f} GB free at {label} ({p}), need ~{need_gb:.1f} GB", flush=True)
        if free_gb < need_gb and (worst is None or free_gb < worst[1]):
            worst = (label, free_gb)
    if worst is not None:
        label, free_gb = worst
        print_disk_hogs()
        raise SystemExit(
            f"NOT ENOUGH DISK at {label} ({free_gb:.1f} < {need_gb:.1f} GB) — free space first:\n"
            "  rm -rf outputs/*/checkpoint-* ~/.cache/huggingface ~/.cache/pip /tmp/*\n"
            "  or redirect caches: export HF_HOME=/kaggle/working/.hf-cache\n"
            "  then rerun. (Checkpoints + HF datasets cache are the usual hogs.)")


def make_training_kwargs(out, bs, accum, epochs, lr, save_steps, save_total_limit,
                         fp16, grad_ckpt, optim, fsdp="", save_only_model=True):
    """Pure dict builder (no transformers import) — unit-testable."""
    kw = dict(output_dir=out, per_device_train_batch_size=bs,
              gradient_accumulation_steps=accum, num_train_epochs=epochs,
              learning_rate=lr, logging_steps=2,
              save_steps=save_steps, save_total_limit=save_total_limit,
              fp16=fp16, report_to="none",
              gradient_checkpointing=grad_ckpt, optim=optim,
              save_only_model=save_only_model)
    if (fsdp or "").strip():
        kw["fsdp"] = fsdp.strip()
        kw["fsdp_config"] = {"fsdp_state_dict_type": "FULL_STATE_DICT",
                             "fsdp_transformer_layer_cls_to_wrap": "LlamaDecoderLayer"}
    return kw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.getenv("ALFA_DATA", "/data/processed/train.jsonl"))
    ap.add_argument("--out", default=os.getenv("ALFA_OUT", "/data/outputs/tiny-50M"))
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--config", default=os.getenv("ALFA_CONFIG", "configs/tiny-50M.yaml"))
    ap.add_argument("--max-seq-len", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--tokenizer", default=os.getenv("ALFA_TOKENIZER", "tokenizers/alfa-32k.json"),
                    help="BPE tokenizer file (missing -> char-level fallback)")
    ap.add_argument("--grad-accum", type=int, default=None,
                    help="override config gradient accumulation (higher = less VRAM per step)")
    ap.add_argument("--optim", default=None,
                    help="optimizer: adamw_torch (default) or adamw_8bit (half optimizer VRAM, needs bitsandbytes)")
    ap.add_argument("--resume-from", default=os.getenv("ALFA_RESUME", ""),
                    help="continue training from this checkpoint dir instead of random init (self-learn cycles)")
    ap.add_argument("--fsdp", default=os.getenv("ALFA_FSDP", ""),
                    help='FSDP sharding across GPUs, e.g. "full_shard" (Kaggle T4x2). Empty = single-GPU.')
    ap.add_argument("--save-steps", type=int, default=None,
                    help="checkpoint every N steps (fewer saves = less disk; default 50)")
    ap.add_argument("--save-total-limit", type=int, default=None,
                    help="keep at most N checkpoints (default 1)")
    ap.add_argument("--resume-ckpt", default="",
                    help="Trainer resume_from_checkpoint path: continues optimizer+step after a crash")
    ap.add_argument("--min-disk-gb", type=float, default=None,
                    help="override disk preflight need (GB). Use when YOU judge space is fine, "
                         "e.g. --min-disk-gb 15. Your risk: mid-run ENOSPC kills the run.")
    ap.add_argument("--save-only-model", dest="save_only_model",
                    action=argparse.BooleanOptionalAction, default=True,
                    help="checkpoints hold weights only (no 9GB optimizer states). "
                         "--no-save-only-model keeps full resume state. Crash resume then uses --resume-from (weights).")
    args = ap.parse_args()

    import yaml
    import torch
    from bpe import char_encode, encode_text, load_bpe_tokenizer
    from transformers import LlamaConfig, LlamaForCausalLM, Trainer, TrainingArguments
    from torch.utils.data import Dataset

    cfg, ctx, trn, mm, mdl = {}, {}, {}, {}, {}
    if args.config and Path(args.config).exists():
        cfg = yaml.safe_load(open(args.config, encoding="utf-8")) or {}
        mdl, ctx, trn, mm = (cfg.get("model", {}), cfg.get("context", {}),
                             cfg.get("training", {}), cfg.get("multimodal", {}))
    seq_len = args.max_seq_len or int(ctx.get("max_seq_len", mdl.get("max_seq_len", 8192)))
    rope_theta = float(mdl.get("rope_theta", 500000.0))
    bs = args.batch_size or int(trn.get("batch_size", 1))
    accum = int(trn.get("grad_accum", trn.get("gradient_accumulation_steps", 8)))
    lr = float(trn.get("lr", 3e-4))
    grad_ckpt = bool(ctx.get("gradient_checkpointing", True))
    truncation = ctx.get("truncation", "right")
    video_frames = int(mm.get("video_frames", 4))
    if args.grad_accum:
        accum = args.grad_accum
    optim = args.optim or trn.get("optim", "adamw_torch")
    fsdp, optim = resolve_fsdp_optim(args.fsdp or trn.get("fsdp", ""), optim)
    if optim == "adamw_8bit":
        try:
            import bitsandbytes  # noqa: F401
        except ImportError:
            raise SystemExit("adamw_8bit needs bitsandbytes: run  pip install bitsandbytes")
    print(f"optim={optim} (8-bit halves optimizer VRAM; use it for 1B on T4)")

    print(f"config: seq_len={seq_len} rope={rope_theta} bs={bs} accum={accum} ckpt={grad_ckpt}")

    bpe_tok = load_bpe_tokenizer(args.tokenizer)
    use_bpe = bpe_tok is not None

    class JsonlDS(Dataset):
        def __init__(self, path, tok_len=8192):
            rows = [json.loads(l) for l in open(path, encoding="utf-8")]
            self.rows, self.tok_len = rows, tok_len
            self.budget = tok_len * 4

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, i):
            r = self.rows[i]
            pre = []
            if r.get("images"):
                pre += ["<image>"] * min(2, len(r["images"]))
            if r.get("video"):
                pre += ["<video>"] * video_frames
            s = (" ".join(pre) + "\n" if pre else "") + (
                r.get("prompt", "") + "\n" + r.get("think", "") + "\n" + r.get("answer", ""))
            if use_bpe:
                ids, attn = encode_text(s, bpe_tok, self.tok_len, truncation)
            else:
                if len(s) > self.budget:
                    s = s[:self.budget] if truncation == "right" else s[-self.budget:]
                ids, attn = char_encode(s, self.tok_len)
            return {"input_ids": torch.tensor(ids, dtype=torch.long),
                    "attention_mask": torch.tensor(attn, dtype=torch.long),
                    "labels": torch.tensor(ids, dtype=torch.long)}

    print(f"data: {args.data} exists={Path(args.data).exists()}")
    cuda = torch.cuda.is_available()
    print(f"torch={torch.__version__} cuda={cuda}", flush=True)
    if cuda:
        print(f"gpu: {torch.cuda.get_device_name(0)} "
              f"({torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB)", flush=True)
    else:
        from bpe import cuda_hint
        print(cuda_hint(), flush=True)
    model_cfg = LlamaConfig(
        hidden_size=int(mdl.get("hidden_size", 512)),
        num_hidden_layers=int(mdl.get("num_layers", 8)),
        num_attention_heads=int(mdl.get("num_heads", 8)),
        num_key_value_heads=int(mdl.get("num_kv_heads", mdl.get("num_heads", 8))),
        intermediate_size=int(mdl.get("intermediate_size", 1376)),
        vocab_size=int(mdl.get("vocab_size", 32000)),
        max_position_embeddings=seq_len,
        rope_theta=rope_theta,
    )
    model = build_model(model_cfg, args.resume_from)
    if grad_ckpt:
        model.gradient_checkpointing_enable()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"params: {n_params / 1e9:.2f}B", flush=True)
    full_ckpt = not args.save_only_model
    if not full_ckpt:
        print("checkpoints: weights-only (no 9GB optimizer states) — crash resume via --resume-from (weights)", flush=True)
    need_gb = args.min_disk_gb if args.min_disk_gb else estimate_need_gb(
        n_params, fp16=torch.cuda.is_available(), full_ckpt=full_ckpt)
    if args.min_disk_gb:
        print(f"disk preflight overridden by user: need ~{need_gb:.1f} GB (your risk)", flush=True)
    check_disk_gb(args.out, need_gb)
    ds = JsonlDS(args.data, tok_len=seq_len)
    kw = make_training_kwargs(
        args.out, bs, accum, args.epochs, lr,
        args.save_steps or 50, args.save_total_limit or 1,
        torch.cuda.is_available(), grad_ckpt, optim, fsdp,
        save_only_model=not full_ckpt)
    if fsdp:
        print(f"FSDP on: {fsdp} (multi-GPU sharding, ~1.5-1.8x faster on 2xT4)", flush=True)
    targs = TrainingArguments(**kw)
    if args.resume_ckpt and not full_ckpt:
        print("NOTE: --resume-ckpt with weights-only checkpoints resumes weights "
              "with a FRESH optimizer (no saved states) — equivalent to --resume-from here.", flush=True)
    trainer = Trainer(model=model, args=targs, train_dataset=ds)
    try:
        trainer.train(
            resume_from_checkpoint=args.resume_ckpt or None)
    except torch.cuda.OutOfMemoryError:
        import traceback
        traceback.print_exc()
        print("\nCUDA OUT OF MEMORY — fix, cheapest first:", flush=True)
        print(" 1. Restart the Colab runtime (old processes may hold VRAM), then rerun.", flush=True)
        print(" 2. Lower memory: --batch-size 1 --grad-accum 16 --max-seq-len 1024", flush=True)
        print(" 3. For 1B on T4: --optim adamw_8bit  (pip install bitsandbytes)", flush=True)
        print(" 4. Still OOM: train tiny-50M instead of 1B on free T4.", flush=True)
        print(" 5. Fragmentation: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True", flush=True)
        raise SystemExit(3)
    # NOTE: must go through trainer.save_model(), NOT model.save_pretrained().
    # Under FSDP the model params are sharded with invalid storages on this
    # process; a direct save crashes in safetensors (data pointer error).
    # trainer.save_model() gathers FULL_STATE_DICT first (same path Trainer
    # itself uses for intermediate checkpoints — proven working: crashed runs
    # reached 100% through all step-N saves and only died on the manual save).
    # rope_scaling must be set BEFORE save: config.json is written by
    # save_model() and it is what activates YaRN at inference time
    # (training itself stays on plain RoPE — scaling only lives in config).
    _apply_rope_scaling(model, mdl.get("rope_scaling"), seq_len)
    trainer.save_model()
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
