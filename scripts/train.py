"""Tiny Llama pretrain demo (CPU/4GB friendly). For 1B+ use cloud + DeepSpeed.

High-context + multimodal aware:
  - reads configs/tiny-50M.yaml (or --config) for max_seq_len / rope_theta
  - prepends <image>/<video> placeholders so rows with images/video keep
    their token budget instead of being silently truncated
  - packs prompt+think+answer with right/left truncation per config

  python scripts/train.py --config configs/tiny-50M.yaml
  python scripts/train.py --max-seq-len 2048 --batch-size 2   # short-ctx local test
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bpe import char_encode, encode_text, load_bpe_tokenizer

import torch
import yaml
from torch.utils.data import Dataset
from transformers import LlamaConfig, LlamaForCausalLM, Trainer, TrainingArguments

DEFAULT_CONFIG = Path("configs/tiny-50M.yaml")


def load_cfg(path: str | None) -> dict:
    cfg = {}
    if path and Path(path).exists():
        cfg = yaml.safe_load(open(path, encoding="utf-8")) or {}
    m = cfg.get("model", {})
    c = cfg.get("context", {})
    t = cfg.get("training", {})
    mm = cfg.get("multimodal", {})
    return {
        "max_seq_len": int(c.get("max_seq_len", m.get("max_seq_len", 8192))),
        "rope_theta": float(m.get("rope_theta", 500000.0)),
        "hidden_size": int(m.get("hidden_size", 512)),
        "num_layers": int(m.get("num_layers", 8)),
        "num_heads": int(m.get("num_heads", 8)),
        "num_kv_heads": int(m.get("num_kv_heads", m.get("num_heads", 8))),
        "intermediate_size": int(m.get("intermediate_size", 1376)),
        "vocab_size": int(m.get("vocab_size", 32000)),
        "image_tokens": int(mm.get("image_tokens", 256)),
        "video_frames": int(mm.get("video_frames", 4)),
        "video_tokens_per_frame": int(mm.get("video_tokens_per_frame", 64)),
        "truncation": c.get("truncation", "right"),
        "grad_ckpt": bool(c.get("gradient_checkpointing", True)),
        "batch_size": int(t.get("batch_size", 1)),
        "grad_accum": int(t.get("grad_accum", t.get("gradient_accumulation_steps", 8))),
        "lr": float(t.get("lr", 3e-4)),
        "epochs": int(t.get("epochs", 1)),
        "output_dir": t.get("output_dir", "outputs/tiny-50M"),
    }


def build_text(row: dict, image_tokens: int, video_frames: int,
               video_tpp: int) -> str:
    """Assemble training text with multimodal placeholder prefixes."""
    prefix = []
    if row.get("images"):
        # one <image> block per image (placeholder count handled as chars,
        # real vision tower wiring is a later step)
        for _ in row["images"][:2]:
            prefix.append("<image>")
    if row.get("video"):
        prefix.append(" ".join(["<video>"] * video_frames))
    if row.get("modality", "").startswith("image") and not row.get("images"):
        # rows that lost their URL still keep budget for the visual tokens
        prefix.append("<image>")
    body = (row.get("prompt", "") + "\n"
            + row.get("think", "") + "\n"
            + row.get("answer", ""))
    pre = (" ".join(prefix) + "\n") if prefix else ""
    return pre + body


class JsonlDS(Dataset):
    def __init__(self, path, tok_len=8192, truncation="right",
                 image_tokens=256, video_frames=4, video_tpp=64,
                 tokenizer_path="tokenizers/alfa-32k.json"):
        self.rows = [json.loads(l) for l in open(path, encoding="utf-8")]
        self.tok_len = tok_len
        # ~4 chars per token heuristic for char-level demo tokenization
        self.char_budget = tok_len * 4
        self.truncation = truncation
        self.image_tokens = image_tokens
        self.video_frames = video_frames
        self.video_tpp = video_tpp
        self.tok = load_bpe_tokenizer(tokenizer_path)
        self.use_bpe = self.tok is not None

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        s = build_text(r, self.image_tokens, self.video_frames, self.video_tpp)
        if self.use_bpe:
            ids, attn = encode_text(s, self.tok, self.tok_len, self.truncation)
        else:
            if len(s) > self.char_budget:
                s = s[:self.char_budget] if self.truncation == "right" else s[-self.char_budget:]
            ids, attn = char_encode(s, self.tok_len)
        return {"input_ids": torch.tensor(ids, dtype=torch.long),
                "attention_mask": torch.tensor(attn, dtype=torch.long),
                "labels": torch.tensor(ids, dtype=torch.long)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--data", default="data/processed/train.jsonl")
    ap.add_argument("--max-seq-len", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--tokenizer", default="tokenizers/alfa-32k.json",
                    help="BPE tokenizer file (missing -> char-level fallback)")
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    if args.max_seq_len:
        cfg["max_seq_len"] = args.max_seq_len
    if args.batch_size:
        cfg["batch_size"] = args.batch_size
    if args.epochs is not None:
        cfg["epochs"] = args.epochs
    if args.out:
        cfg["output_dir"] = args.out

    print(f"config: seq_len={cfg['max_seq_len']} rope_theta={cfg['rope_theta']} "
          f"bs={cfg['batch_size']} accum={cfg['grad_accum']} ckpt={cfg['grad_ckpt']}")
    cuda = torch.cuda.is_available()
    print(f"torch={torch.__version__} cuda={cuda}", flush=True)
    if cuda:
        print(f"gpu: {torch.cuda.get_device_name(0)}", flush=True)
    else:
        print("WARNING: CUDA not visible — training will run on CPU (~10x slower). "
              "On Colab: Runtime > Change runtime type > T4 GPU, then restart + rerun.", flush=True)

    model_cfg = LlamaConfig(
        hidden_size=cfg["hidden_size"], num_hidden_layers=cfg["num_layers"],
        num_attention_heads=cfg["num_heads"],
        num_key_value_heads=cfg["num_kv_heads"],
        intermediate_size=cfg["intermediate_size"],
        vocab_size=cfg["vocab_size"],
        max_position_embeddings=cfg["max_seq_len"],
        rope_theta=cfg["rope_theta"],
    )
    model = LlamaForCausalLM(model_cfg)
    if cfg["grad_ckpt"]:
        model.gradient_checkpointing_enable()

    ds = JsonlDS(args.data, tok_len=cfg["max_seq_len"],
                 truncation=cfg["truncation"],
                 image_tokens=cfg["image_tokens"],
                 video_frames=cfg["video_frames"],
                 video_tpp=cfg["video_tokens_per_frame"],
                 tokenizer_path=args.tokenizer)
    targs = TrainingArguments(
        output_dir=cfg["output_dir"],
        per_device_train_batch_size=cfg["batch_size"],
        gradient_accumulation_steps=cfg["grad_accum"],
        num_train_epochs=cfg["epochs"],
        learning_rate=cfg["lr"],
        logging_steps=5, save_steps=50,
        fp16=torch.cuda.is_available(),
        gradient_checkpointing=cfg["grad_ckpt"],
        report_to="none",
    )
    Trainer(model=model, args=targs, train_dataset=ds).train()
    model.save_pretrained(cfg["output_dir"])
    print(f"saved -> {cfg['output_dir']}")


if __name__ == "__main__":
    main()
