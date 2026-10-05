"""Test inference with a trained tiny-50M checkpoint.

The model was trained with char-level pseudo-tokenization
(ids = ord(char) capped at 30000), so inference MUST use the same
encoding — the BPE tokenizer in tokenizers/ is NOT compatible
with these weights (wiring BPE is future work).

Sources:
  Hub (private repo, needs HF_TOKEN env):
    python scripts/infer.py --repo mrsandip/Alfa-Code --subfolder outputs/tiny-50M \\
        --prompt "def fib(n):" --max-new-tokens 200
  Local dir:
    python scripts/infer.py --local /content/outputs/tiny-50M --prompt "def fib(n):"

Expects: real (demo-quality) continuation text. Tiny 50M + 1 epoch +
char-level encoding => proves the pipeline, not SOTA code quality.
"""
import argparse
import os
from pathlib import Path

import torch
from transformers import LlamaForCausalLM

VOCAB = 32000


def encode(text: str, max_len: int = 2048) -> torch.Tensor:
    ids = [min(ord(c), VOCAB - 1) for c in text][:max_len]
    return torch.tensor([ids], dtype=torch.long)


def decode(ids) -> str:
    out = []
    for i in ids:
        i = int(i)
        if i in (9, 10, 13) or 32 <= i < 0x110000:
            try:
                out.append(chr(i))
            except ValueError:
                pass
    return "".join(out)


def resolve_ckpt(args) -> str:
    if args.local:
        p = Path(args.local)
        assert (p / "config.json").exists(), f"no config.json in {p}"
        return str(p)
    from huggingface_hub import snapshot_download

    tok = os.getenv("HF_TOKEN")
    if not tok:
        raise SystemExit("HF_TOKEN env not set — needed for the private Hub repo. "
                         "Use --local PATH for a local checkpoint instead.")
    d = snapshot_download(args.repo, allow_patterns=[f"{args.subfolder}/*"], token=tok)
    p = Path(d) / args.subfolder
    assert (p / "config.json").exists(), f"no config.json in {p}"
    return str(p)


def main():
    ap = argparse.ArgumentParser(description="Tiny-50M inference test")
    ap.add_argument("--repo", default="mrsandip/Alfa-Code")
    ap.add_argument("--subfolder", default="outputs/tiny-50M")
    ap.add_argument("--local", default=None)
    ap.add_argument("--prompt", default="def fib(n):")
    ap.add_argument("--wrap", action="store_true",
                    help="wrap prompt like training rows (Complete and explain...)")
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    ckpt = resolve_ckpt(args)
    print(f"loading {ckpt} ...", flush=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = LlamaForCausalLM.from_pretrained(
        ckpt, torch_dtype=torch.float16 if device == "cuda" else torch.float32)
    model.to(device).eval()
    print(f"model on {device}, params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M",
          flush=True)

    text = args.prompt
    if args.wrap:
        text = f"Complete and explain this file:\n<code>\n{args.prompt}\n</code>"
    inp = encode(text).to(device)
    print(f"prompt tokens: {inp.shape[1]}", flush=True)

    with torch.no_grad():
        out = model.generate(
            inp, max_new_tokens=args.max_new_tokens,
            do_sample=True, temperature=args.temperature, top_p=args.top_p,
            pad_token_id=0,
        )
    gen = decode(out[0][inp.shape[1]:])
    print("=" * 60)
    print("PROMPT:")
    print(text)
    print("=" * 60)
    print("GENERATED:")
    print(gen)
    print("=" * 60)


if __name__ == "__main__":
    main()
