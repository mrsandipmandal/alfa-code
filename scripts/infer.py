"""Test inference with a trained tiny-50M checkpoint.

Encoding MUST match training: BPE via tokenizers/alfa-32k.json when the
checkpoint was BPE-trained, else legacy char-level fallback. Mixing them
(char-trained weights + BPE ids or vice versa) gives garbage.

Sources:
  Hub (private repo, needs HF_TOKEN env):
    python scripts/infer.py --repo mrsandip/Alfa-Code --subfolder outputs/tiny-50M \\
        --prompt "def fib(n):" --max-new-tokens 200
  Local dir:
    python scripts/infer.py --local /content/outputs/tiny-50M --prompt "def fib(n):"
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bpe import char_encode, decode_ids, encode_text, load_bpe_tokenizer

import torch
from transformers import LlamaForCausalLM


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
    ap.add_argument("--tokenizer", default="tokenizers/alfa-32k.json",
                    help="BPE tokenizer file (missing -> char-level fallback)")
    ap.add_argument("--max-prompt-tokens", type=int, default=1024)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    tok = load_bpe_tokenizer(args.tokenizer)
    print(f"encoding: {'BPE' if tok is not None else 'CHAR fallback'}", flush=True)
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
    if tok is not None:
        ids, _ = encode_text(text, tok, args.max_prompt_tokens, pad=False)
    else:
        ids, _ = char_encode(text, args.max_prompt_tokens)
    inp = torch.tensor([ids], dtype=torch.long).to(device)
    print(f"prompt tokens: {inp.shape[1]}", flush=True)

    pad_id = tok.pad_token_id if (tok is not None and tok.pad_token_id is not None) else 0
    eos_id = tok.eos_token_id if tok is not None else None
    with torch.no_grad():
        out = model.generate(
            inp, max_new_tokens=args.max_new_tokens,
            do_sample=True, temperature=args.temperature, top_p=args.top_p,
            pad_token_id=pad_id, eos_token_id=eos_id,
        )
    gen = decode_ids(out[0][inp.shape[1]:].tolist(), tok)
    print("=" * 60)
    print("PROMPT:")
    print(text)
    print("=" * 60)
    print("GENERATED:")
    print(gen)
    print("=" * 60)


if __name__ == "__main__":
    main()
