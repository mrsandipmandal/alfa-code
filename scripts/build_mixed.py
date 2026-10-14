"""Build the mixed-1.2 SFT dataset: normalize -> quality gate -> split.

Inputs (any mix):
  - flat rows from download_data.py:
      {modality, prompt, think, answer, ...}
  - messages rows (hand-written seeds):
      {messages: [{role, content}, ...], source?}
  - leetcode rows may carry {tests, entry_point} for the executable gate

Output (JSONL, one messages object per line):
  {"messages": [{"role": "user", ...}, {"role": "assistant", ...}], "source": ...}
  + data/processed/mixed-1.2-{train,valid}.jsonl (95/5, seed-fixed)

Quality gates (counts printed at the end):
  - eval-lock: modality == "eval" rows are REFUSED (never train on eval)
  - shape: >=1 user + >=1 assistant turn, all contents non-empty strings
  - length: answer >= --min-answer-chars, total <= --max-total-chars
  - code gate: python-fenced answers must ast.parse (first ```python block)
  - danger DROP (regex): os.system / shell=True / rm -rf / reboot / fork-bomb...
  - secret DROP (regex): api keys, passwords, hf_/ghp_/AKIA tokens, .env blobs
  - dedup: exact sha256 on normalized messages + near-dup head key

Example:
  python scripts/build_mixed.py \\
      --inputs data/raw/code_feedback.jsonl data/raw/python_codes.jsonl \\
      --seeds data/seeds/bengali-qa.jsonl data/seeds/project-qa.jsonl \\
      --out-train data/processed/mixed-1.2-train.jsonl \\
      --out-valid data/processed/mixed-1.2-valid.jsonl
"""
import argparse
import ast
import hashlib
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

# ---- drop patterns (case-insensitive, searched in prompt+answer) ----
DANGER_RES = [
    r"os\s*\.\s*system\s*\(",
    r"subprocess\s*\.\s*(call|run|Popen)[^)]*shell\s*=\s*True",
    r"\brm\s+-rf\b",
    r"\btaskkill\b",
    r"\bshutdown\b|\breboot\b|\bhalt\b|\bpoweroff\b",
    r"\bmkfs\b|\bdd\s+if=",
    r":\(\)\s*\{\s*:",
    r"EmptyRecycleBin|SystemParametersInfoW",
]
DANGER_RE = re.compile("|".join(f"(?:{p})" for p in DANGER_RES), re.IGNORECASE)

SECRET_RES = [
    r"api[_-]?key\s*[:=]\s*['\"][A-Za-z0-9_\-]{8,}['\"]",
    r"password\s*[:=]\s*['\"][^'\"]{4,}['\"]",
    r"\bhf_[A-Za-z0-9]{20,}",
    r"\bghp_[A-Za-z0-9]{20,}",
    r"\bAKIA[0-9A-Z]{16}\b",
    r"xox[bap]-",
]
SECRET_RE = re.compile("|".join(f"(?:{p})" for p in SECRET_RES))

FENCE_RE = re.compile(r"```(?:python|py)\n(.*?)```", re.DOTALL)


def to_messages(row: dict):
    """Flat row OR messages row -> (messages, source, modality, extra)."""
    if isinstance(row.get("messages"), list) and row["messages"]:
        msgs = [{"role": str(m.get("role", "")).lower(),
                 "content": str(m.get("content", ""))}
                for m in row["messages"] if isinstance(m, dict)]
        extra = {k: row[k] for k in ("tests", "entry_point", "task_id")
                 if k in row}
        return msgs, row.get("source", "messages"), row.get("modality", ""), extra
    prompt = str(row.get("prompt", "") or "")
    think = str(row.get("think", "") or "")
    answer = str(row.get("answer", "") or "")
    user = prompt + ("\n" + think if think.strip() else "")
    msgs = [{"role": "user", "content": user},
            {"role": "assistant", "content": answer}]
    extra = {k: row[k] for k in ("tests", "entry_point", "task_id")
             if k in row}
    return msgs, row.get("source", "flat"), row.get("modality", ""), extra


def gate(msgs, source, modality, extra, args, stats):
    stats["in"] += 1
    if modality == "eval":
        print("FATAL: eval-locked row reached the trainer input "
              f"(source={source}). Refusing.", file=sys.stderr)
        sys.exit(2)
    roles = [m["role"] for m in msgs]
    if "user" not in roles or "assistant" not in roles:
        stats["drop:shape"] += 1
        return None
    if any(not isinstance(m["content"], str) or not m["content"].strip()
           for m in msgs):
        stats["drop:empty"] += 1
        return None
    blob = "\n".join(m["content"] for m in msgs)
    asst = " ".join(m["content"] for m in msgs if m["role"] == "assistant")
    if len(asst.strip()) < args.min_answer_chars:
        stats["drop:short"] += 1
        return None
    if len(blob) > args.max_total_chars:
        stats["drop:toolong"] += 1
        return None
    if DANGER_RE.search(blob):
        stats["drop:danger"] += 1
        return None
    if SECRET_RE.search(blob):
        stats["drop:secret"] += 1
        return None
    if modality.startswith("code") or "```python" in asst or "```py" in asst:
        m = FENCE_RE.search(asst)
        if m:
            try:
                ast.parse(m.group(1))
            except SyntaxError:
                stats["drop:badsyntax"] += 1
                return None
    norm = json.dumps(msgs, ensure_ascii=False, sort_keys=True)
    digest = hashlib.sha256(norm.encode()).hexdigest()
    if digest in gate.seen_exact:
        stats["drop:dup-exact"] += 1
        return None
    gate.seen_exact.add(digest)
    head = (msgs[0]["content"][:256],
            next((m["content"][:256] for m in msgs
                  if m["role"] == "assistant"), ""))
    if head in gate.seen_head:
        stats["drop:dup-near"] += 1
        return None
    gate.seen_head.add(head)
    stats["kept"] += 1
    stats[f"src:{source}"] += 1
    return {"messages": msgs, "source": source}


gate.seen_exact = set()
gate.seen_head = set()


def read_rows(paths):
    for p in paths:
        if not Path(p).exists():
            print(f"[warn] missing input, skipping: {p}")
            continue
        with open(p, encoding="utf-8") as f:
            for ln, line in enumerate(f, 1):
                if not line.strip():
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    print(f"[warn] bad JSON {p}:{ln}, skipping")


def main():
    ap = argparse.ArgumentParser(description="mixed-1.2 dataset builder")
    ap.add_argument("--inputs", nargs="*", default=[],
                    help="flat-row jsonl files (download_data.py output)")
    ap.add_argument("--seeds", nargs="*", default=[],
                    help="hand-written messages jsonl files (missing = warn+skip)")
    ap.add_argument("--out-train", required=True)
    ap.add_argument("--out-valid", required=True)
    ap.add_argument("--valid-frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min-answer-chars", type=int, default=20)
    ap.add_argument("--max-total-chars", type=int, default=12000)
    args = ap.parse_args()

    stats = Counter()
    kept = []
    for row in read_rows(args.inputs + args.seeds):
        msgs, source, modality, _extra = to_messages(row)
        out = gate(msgs, source, modality, _extra, args, stats)
        if out is not None:
            kept.append(out)

    random.Random(args.seed).shuffle(kept)
    n_valid = max(1, int(len(kept) * args.valid_frac)) if kept else 0
    valid, train = kept[:n_valid], kept[n_valid:]
    for path, rows in ((args.out_train, train), (args.out_valid, valid)):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"in={stats['in']} kept={stats['kept']} "
          f"train={len(train)} valid={len(valid)}")
    for k in sorted(stats):
        if k.startswith(("drop:", "src:")):
            print(f"  {k}: {stats[k]}")
    print(f"wrote {args.out_train} + {args.out_valid}")


if __name__ == "__main__":
    main()
