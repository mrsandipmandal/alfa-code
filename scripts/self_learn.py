"""Self-learning loop, step 1: generate -> filter -> merge.

Uses the current checkpoint to generate new Q&A/code rows from seeds
(topics file + sampled train prompts), filters them through quality
gates, and appends survivors to the training file (synthetic cap 30%).

Full cycle (Colab, 1B manual):
  !python scripts/self_learn.py --cycle 1 --topics data/seeds/topics.txt \\
      --from-train 100 --n 200 --max-tokens 256 --append
  # then re-run the train cell with --out /content/outputs/self-cycle-1
  # review the cycle report, then promote: cp -r to base-1B + upload cell

Gates (abort or trim, never silently poison data):
  - meaningful: >=8 word chars (same rule as UI backend)
  - dedup: normalized match vs train file AND within batch
  - code-parse: code-looking answers must ast.parse (python) 
  - keep-rate: <20% kept -> abort, nothing appended
  - synth-cap: synthetic rows stay <=30% of the merged file
"""
import argparse
import ast
import json
import os
import random
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "app"))
from bpe import char_encode, decode_ids, encode_text, load_bpe_tokenizer  # noqa: E402


def meaningful(t: str) -> bool:
    return len(re.sub(r"\W", "", t or "")) >= 8


def norm(t: str) -> str:
    return re.sub(r"\W", "", (t or "").lower())


def looks_code(prompt: str, answer: str) -> bool:
    blob = f"{prompt}\n{answer}"
    return any(k in blob for k in ("def ", "class ", "import ", "<code>", "```", "return "))


def looks_python(prompt: str, answer: str) -> bool:
    """Only Python-looking rows face the ast gate (HTML/JS never parses)."""
    blob = f"{prompt}\n{answer}"
    if "<" in blob and ">" in blob:
        return False  # markup, not python
    return bool(re.search(r"\b(def |class |import |return )", blob))


def extract_code(gen: str):
    """Best parseable python chunk: fenced blocks first, then whole text."""
    fences = re.findall(r"```(?:\w+)?\n([\s\S]*?)```", gen)
    for cand in fences + [gen]:
        try:
            ast.parse(cand.strip())
            return cand.strip()
        except SyntaxError:
            continue
    return None


def clean_answer(a: str) -> str:
    a = (a or "").strip()
    m = re.match(r"```(?:\w+)?\n([\s\S]*?)```", a)
    if m:
        a = m.group(1).strip()
    return a[:12000]


def load_seeds(topics_file, from_train, n, train_file, seed):
    rng = random.Random(seed)
    seeds = []
    if topics_file and Path(topics_file).exists():
        for line in open(topics_file, encoding="utf-8"):
            line = line.strip()
            if line and not line.startswith("#"):
                seeds.append(line)
    if from_train > 0 and Path(train_file).exists():
        rows = [json.loads(l) for l in open(train_file, encoding="utf-8")]
        rng.shuffle(rows)
        for r in rows[:from_train]:
            p = (r.get("prompt", "") or "").strip()
            if len(p) > 20:
                seeds.append(p)
    elif from_train > 0:
        print(f"WARNING: --from-train {from_train} but train file missing: {train_file} (train seeds skipped)")
    rng.shuffle(seeds)
    return seeds[:n] if n > 0 else seeds


def existing_norms(train_file):
    norms = set()
    if Path(train_file).exists():
        for line in open(train_file, encoding="utf-8"):
            try:
                r = json.loads(line)
                norms.add(norm(r.get("prompt", "")) + "|" + norm(r.get("answer", "")))
            except Exception:
                continue
    return norms


def main():
    ap = argparse.ArgumentParser(description="Self-learning data generation")
    ap.add_argument("--cycle", type=int, default=1)
    ap.add_argument("--ckpt", default=os.getenv("ALFA_MODEL", "outputs/tiny-50M"))
    ap.add_argument("--tokenizer", default=os.getenv("ALFA_TOKENIZER", "tokenizers/alfa-32k.json"))
    ap.add_argument("--topics", default="data/seeds/topics.txt")
    ap.add_argument("--from-train", type=int, default=100)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--train-file", default="data/processed/train.jsonl")
    ap.add_argument("--out-dir", default="data/synth")
    ap.add_argument("--append", action="store_true", help="append survivors to train file (cap 30%%)")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--min-keep-rate", type=float, default=0.20)
    ap.add_argument("--max-synth-ratio", type=float, default=0.30)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    import torch
    from transformers import LlamaForCausalLM

    tok = load_bpe_tokenizer(args.tokenizer)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"loading {args.ckpt} ...", flush=True)
    model = LlamaForCausalLM.from_pretrained(
        args.ckpt, torch_dtype=torch.float16 if device == "cuda" else torch.float32)
    model.to(device).eval()

    seeds = load_seeds(args.topics, args.from_train, args.n, args.train_file, args.seed)
    print(f"seeds: {len(seeds)} (topics + train-sampled)", flush=True)
    if not seeds:
        raise SystemExit("no seeds — check --topics and --train-file")

    seen = existing_norms(args.train_file)
    kept, dropped = [], {"weak": 0, "dup": 0, "parse": 0}
    torch.manual_seed(args.seed)
    pad_id = (tok.pad_token_id if (tok is not None and tok.pad_token_id is not None) else 0)

    for i, prompt in enumerate(seeds):
        task = f"Complete and explain this task:\n{prompt}\nAnswer briefly, then show the code."
        if tok is not None:
            ids, _ = encode_text(task, tok, 1024, pad=False)
        else:
            ids, _ = char_encode(task, 1024)
        inp = torch.tensor([ids], dtype=torch.long).to(device)
        ok, gen = False, ""
        for tries in range(3):  # retry warmer, same as UI backend
            with torch.no_grad():
                out = model.generate(
                    inp, max_new_tokens=args.max_tokens, do_sample=True,
                    temperature=min(args.temperature + 0.2 * tries, 1.5),
                    top_p=0.95, repetition_penalty=1.15, pad_token_id=pad_id)
            gen = clean_answer(decode_ids(out[0][inp.shape[1]:].tolist(), tok))
            if meaningful(gen):
                ok = True
                break
        if not ok:
            dropped["weak"] += 1
            continue
        key = norm(prompt) + "|" + norm(gen)
        if key in seen:
            dropped["dup"] += 1
            continue
        if looks_code(prompt, gen):
            if looks_python(prompt, gen):
                code = extract_code(gen)
                if code is None:
                    dropped["parse"] += 1
                    continue
                gen = code
            modality = "code"
        else:
            modality = "chat"
        seen.add(key)
        kept.append({"modality": modality, "prompt": prompt[:24000], "think": "",
                     "answer": gen, "images": [], "video": None,
                     "context_len": len(prompt) + len(gen),
                     "source": f"synth-cycle-{args.cycle}"})
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(seeds)} seeds, kept {len(kept)}", flush=True)

    total = len(seeds)
    rate = len(kept) / max(1, total)
    print(f"generated={total} kept={len(kept)} dropped={dropped} keep-rate={rate:.2f}", flush=True)
    if rate < args.min_keep_rate:
        raise SystemExit(f"ABORT: keep-rate {rate:.2f} < {args.min_keep_rate} — nothing written")

    write_and_merge(kept, args)


def write_and_merge(kept, args):
    """Write synth file + optionally append to train file under synth cap."""
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    synth_path = out_dir / f"cycle-{args.cycle}.jsonl"
    with open(synth_path, "w", encoding="utf-8") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(kept)} rows -> {synth_path}")

    if args.append:
        existing = ([json.loads(l) for l in open(args.train_file, encoding="utf-8")]
                    if Path(args.train_file).exists() else [])
        n_synth_old = sum(1 for r in existing if str(r.get("source", "")).startswith("synth-"))
        # cap: synthetic fraction of merged file <= max_synth_ratio
        room = int((len(existing) + len(kept)) * args.max_synth_ratio) - n_synth_old
        addable = kept[:max(0, room)] if room < len(kept) else kept
        if len(addable) < len(kept):
            print(f"cap: adding {len(addable)}/{len(kept)} (synth ratio cap {args.max_synth_ratio})")
        with open(args.train_file, "a+", encoding="utf-8") as f:
            f.seek(0, 2)  # end: ensure trailing newline or rows glue together
            if f.tell() > 0:
                f.seek(f.tell() - 1)
                if f.read(1) != "\n":
                    f.write("\n")
            for r in addable:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"appended {len(addable)} rows -> {args.train_file}")
    return synth_path


if __name__ == "__main__":
    main()
