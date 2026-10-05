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
        print("WARNING: CUDA not visible — training will run on CPU (~10x slower). "
              "On Colab: Runtime > Change runtime type > T4 GPU, then restart + rerun. "
              "If torch is CPU-only, reinstall with CUDA support.", flush=True)
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
