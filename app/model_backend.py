"""Model backend for the Alfa-Code UI (no gradio dependency, testable).

Loads the trained tiny-50M checkpoint + BPE tokenizer when available,
otherwise falls back to mock mode with a clear reason.

Env:
  ALFA_MODEL     checkpoint dir (default: outputs/tiny-50M)
  ALFA_TOKENIZER BPE file      (default: tokenizers/alfa-32k.json)
"""
import os
import re
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
from bpe import char_encode, decode_ids, encode_text, load_bpe_tokenizer  # noqa: E402

MODEL_DIR = os.getenv("ALFA_MODEL", str(REPO / "outputs" / "tiny-50M"))
TOK_PATH = os.getenv("ALFA_TOKENIZER", str(REPO / "tokenizers" / "alfa-32k.json"))

_state = {}


def get_backend():
    """Load once; return dict(model, tok, device, mode, reason)."""
    if _state:
        return _state
    try:
        import torch
        from transformers import LlamaForCausalLM

        if not Path(MODEL_DIR, "config.json").exists():
            raise FileNotFoundError(
                f"no checkpoint at {MODEL_DIR} (train first or set ALFA_MODEL)")
        tok = load_bpe_tokenizer(TOK_PATH)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = LlamaForCausalLM.from_pretrained(
            MODEL_DIR, torch_dtype=torch.float16 if device == "cuda" else torch.float32)
        model.to(device).eval()
        _state.update(model=model, tok=tok, device=device,
                      mode="real" if tok is not None else "char",
                      reason="" if tok is not None else "BPE missing: char fallback")
    except Exception as e:  # missing ckpt / no torch / OOM -> mock mode
        _state.update(model=None, tok=None, device="cpu", mode="mock", reason=str(e))
    return _state


def reset_backend():
    _state.clear()


def build_prompt(prompt, image=None, file=None, video=None):
    """Same shape as training rows: placeholders + filenames + wrapped task."""
    pre = []
    ctx = []
    if image:
        pre.append("<image>")
        ctx.append(f"image: {Path(image).name}")
    if file:
        ctx.append(f"file: {Path(file).name if isinstance(file, str) else 'uploaded'}")
    if video:
        pre += ["<video>"] * 4
        ctx.append(f"video: {Path(video).name} (4 frames extracted)")
    header = (" ".join(pre) + "\n") if pre else ""
    task = f"Complete and explain this task:\n{header}{prompt}"
    if ctx:
        task += f"\nContext files: {', '.join(ctx)}"
    # question-style prompts (chat Q&A) get an explicit code-answer nudge,
    # since training rows are completion-style, not conversational
    p = (prompt or "").strip().lower()
    if p.endswith("?") or p.startswith(("what ", "how ", "why ", "are you", "is ",
                                        "do you", "can you", "explain", "write")):
        task += "\nAnswer briefly, then show the code."
    return task, ctx


def generate(prompt, image=None, file=None, video=None,
             max_new_tokens=256, temperature=0.6, repetition_penalty=1.15):
    """Yield (status, chat_markdown) steps for the Gradio UI."""
    b = get_backend()
    task, ctx = build_prompt(prompt or "", image, file, video)
    header = f"Context: {', '.join(ctx)}\n" if ctx else ""

    yield "[Thinking...]", f"{header}Understanding: {(prompt or '')[:120]}..."
    time.sleep(0.3)
    yield "[Planning...]", f"{header}Plan: 1.parse 2.generate 3.verify (engine: {b['mode']})"
    time.sleep(0.3)

    if b["mode"] == "mock":
        code = (f"```python\n# mock mode ({b['reason']})\n"
                f"# train + set ALFA_MODEL to enable real generation\n"
                f"def solve():\n    return 'todo'\n```")
        yield "[Coding...]", f"{header}{code}"
    else:
        import torch

        yield "[Coding...]", f"{header}Generating with trained model..."
        if b["tok"] is not None:
            ids, _ = encode_text(task, b["tok"], 1024, pad=False)
        else:
            ids, _ = char_encode(task, 1024)
        inp = torch.tensor([ids], dtype=torch.long).to(b["device"])
        pad_id = (b["tok"].pad_token_id
                  if (b["tok"] is not None and b["tok"].pad_token_id is not None) else 0)

        def _run(temp, seed):
            if seed is not None:
                torch.manual_seed(seed)
            with torch.no_grad():
                out = b["model"].generate(
                    inp, max_new_tokens=max_new_tokens,
                    do_sample=True, temperature=temp, top_p=0.95,
                    repetition_penalty=repetition_penalty,
                    pad_token_id=pad_id)
            return decode_ids(out[0][inp.shape[1]:].tolist(), b["tok"])

        gen = _run(temperature, None)
        tries = 1

        def _meaningful(t: str) -> bool:
            # tiny models sometimes emit EOS/whitespace/a lone punctuation (",", ".")
            return len(re.sub(r"\W", "", t)) >= 8

        # tiny models sometimes emit EOS/whitespace immediately -> retry warmer
        while not _meaningful(gen) and tries < 3:
            tries += 1
            yield "[Coding...]", f"{header}Weak output, retrying ({tries}/3)..."
            gen = _run(min(temperature + 0.2 * tries, 1.5), 1000 + tries)
        if _meaningful(gen):
            code = f"```python\n{gen}\n```\n\n_{len(gen.split())} words generated._"
        else:
            code = ("_No usable output after 3 tries — the tiny model emitted only "
                    "blank/punctuation tokens. Try a code-style prompt (e.g. `def fib(n):`), "
                    "press Generate again, or retrain longer for better results._")
        yield "[Coding...]", f"{header}{code}"

    yield ("[Done]",
           f"{header}<details><summary>Reasoning (engine: {b['mode']})</summary>"
           f"Understand &gt; Plan &gt; Code</details>\n\n{code}")
