"""Shared BPE tokenizer helpers (used by train.py, train_hf.py, infer.py).

Primary path: real BPE via tokenizers/alfa-32k.json (PreTrainedTokenizerFast).
Fallback: legacy char-level pseudo-tokenization (ord(char) capped) so
pipelines never crash when the BPE file is missing — with a loud warning.
"""
from pathlib import Path

VOCAB_SIZE = 32000
DEFAULT_TOKENIZER = Path("tokenizers/alfa-32k.json")

SPECIAL = {
    "unk_token": "<unk>",
    "pad_token": "<pad>",
    "bos_token": "<s>",
    "eos_token": "</s>",
}


def load_bpe_tokenizer(path=None):
    """Return PreTrainedTokenizerFast or None (fallback to char-level)."""
    from transformers import PreTrainedTokenizerFast

    p = Path(path) if path else DEFAULT_TOKENIZER
    if not p.exists():
        print(f"WARNING: BPE tokenizer missing at {p} — falling back to "
              f"char-level encoding (demo quality). Run scripts/train_tokenizer.py first.",
              flush=True)
        return None
    tok = PreTrainedTokenizerFast(tokenizer_file=str(p), **SPECIAL)
    tok.model_max_length = 10 ** 9  # we handle truncation ourselves
    print(f"BPE tokenizer loaded from {p} (vocab={tok.vocab_size})", flush=True)
    return tok


def encode_text(text: str, tok, max_len: int, truncation: str = "right", pad: bool = True):
    """BPE-encode to fixed-length (ids, attention_mask). pad=False for inference prompts."""
    enc = tok(text, add_special_tokens=False, truncation=True,
              max_length=max_len,
              truncation_side=truncation if truncation in ("right", "left") else "right")
    ids = enc["input_ids"]
    mask = enc["attention_mask"]
    if pad:
        pad_n = max_len - len(ids)
        if pad_n > 0:
            pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
            ids = ids + [pad_id] * pad_n
            mask = mask + [0] * pad_n
    return ids, mask


def char_encode(text: str, max_len: int):
    ids = [min(ord(c), VOCAB_SIZE - 1) for c in text][:max_len]
    mask = [1] * len(ids)
    pad = max_len - len(ids)
    return ids + [0] * pad, mask + [0] * pad


def char_decode(ids) -> str:
    out = []
    for i in ids:
        i = int(i)
        if i in (9, 10, 13) or 32 <= i < 0x110000:
            try:
                out.append(chr(i))
            except ValueError:
                pass
    return "".join(out)


def decode_ids(ids, tok) -> str:
    if tok is None:
        return char_decode(ids)
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    pad_id = tok.pad_token_id
    ids = [i for i in ids if i != pad_id]
    return tok.decode(ids, skip_special_tokens=False)


def cuda_hint() -> str:
    """Platform-aware fix-it hint when torch can't see CUDA."""
    import os

    if os.getenv("KAGGLE_KERNEL_RUN_TYPE"):
        where = ("On Kaggle: stop this session, then Session options > "
                 "Accelerator > GPU T4x2 (NOT TPU — our stack is CUDA-only), "
                 "Internet ON, then rerun.")
    else:
        where = ("On Colab: Runtime > Change runtime type > T4 GPU, "
                 "then restart + rerun.")
    return ("WARNING: CUDA not visible — training will run on CPU (~10x slower). "
            f"{where} If torch is CPU-only, reinstall with CUDA support.")
