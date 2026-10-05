"""HF Jobs compatible tiny-train: reads /data, writes /data/outputs + Hub.

High-context + multimodal aware (mirrors scripts/train.py):
  python scripts/train_hf.py --data data/processed/train.jsonl \\
      --out /content/outputs/tiny-50M --epochs 1 --config configs/tiny-50M.yaml
"""
import argparse
import json
import os
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.getenv("ALFA_DATA", "/data/processed/train.jsonl"))
    ap.add_argument("--out", default=os.getenv("ALFA_OUT", "/data/outputs/tiny-50M"))
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--config", default=os.getenv("ALFA_CONFIG", "configs/tiny-50M.yaml"))
    ap.add_argument("--max-seq-len", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    args = ap.parse_args()

    import yaml
    import torch
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

    print(f"config: seq_len={seq_len} rope={rope_theta} bs={bs} accum={accum} ckpt={grad_ckpt}")

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
            if len(s) > self.budget:
                s = s[:self.budget] if truncation == "right" else s[-self.budget:]
            ids = [min(ord(c), 30000) for c in s][:self.tok_len]
            attn = [1] * len(ids)
            pad = self.tok_len - len(ids)
            ids += [0] * pad
            attn += [0] * pad
            return {"input_ids": torch.tensor(ids, dtype=torch.long),
                    "attention_mask": torch.tensor(attn, dtype=torch.long),
                    "labels": torch.tensor(ids, dtype=torch.long)}

    print(f"data: {args.data} exists={Path(args.data).exists()}")
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
    model = LlamaForCausalLM(model_cfg)
    if grad_ckpt:
        model.gradient_checkpointing_enable()
    ds = JsonlDS(args.data, tok_len=seq_len)
    targs = TrainingArguments(
        output_dir=args.out, per_device_train_batch_size=bs,
        gradient_accumulation_steps=accum,
        num_train_epochs=args.epochs, learning_rate=lr,
        logging_steps=2, save_steps=50, save_total_limit=1,
        fp16=torch.cuda.is_available(), report_to="none",
        gradient_checkpointing=grad_ckpt,
    )
    Trainer(model=model, args=targs, train_dataset=ds).train()
    model.save_pretrained(args.out)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
