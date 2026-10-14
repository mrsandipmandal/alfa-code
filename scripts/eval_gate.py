"""Phase-D eval gate for alfa-1.2: LeetCode pass@1 vs a running GGUF endpoint.

Usage (model must already serve, e.g. llama-server -m x.gguf -c 2048):
  python scripts/eval_gate.py --endpoint http://127.0.0.1:8080 \\
      --test-file data/processed/leetcode-test.jsonl --n-tasks 20 \\
      --out reports/eval-1.2-exp1.json

Per task: problem_description -> /completion (temp 0) -> extract first
```python fence (else whole output) -> exec with std preamble in a
subprocess (15s timeout; os/sys/subprocess imports = auto-fail) ->
run the row's `test` asserts against `entry_point` -> pass/fail.

Also (optional): --eval-log <merged/eval_log.json> folds valid loss/ppl
into the report; --seeds-bn <bengali-qa.jsonl> runs 10 spot prompts whose
outputs need HUMAN grading (auto-grade impossible) -> report lists them.

Gate: PASS iff pass@1 >= --min-pass (default 0.10 — LeetCode is Hard;
tune per iteration). Exit 0 on PASS, 1 on FAIL. NEVER train on the
test file (see build_mixed.py eval lock).
"""
import argparse
import json
import re
import subprocess
import sys
import tempfile
import os
from pathlib import Path

import urllib.request

FENCE_RE = re.compile(r"```(?:python|py)\n(.*?)```", re.DOTALL)
PREAMBLE = "from typing import *\n"
BANNED = re.compile(r"import\s+(os|sys|subprocess|socket|shutil|pathlib)|"
                    r"os\s*\.\s*system|open\s*\(", re.IGNORECASE)


def api_post(endpoint, path, payload, timeout=300):
    # ensure_ascii: old llama-server builds 500 on raw non-ASCII bodies
    body = json.dumps(payload, ensure_ascii=True).encode()
    req = urllib.request.Request(
        endpoint.rstrip("/") + path, data=body,
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def extract_code(text):
    m = FENCE_RE.search(text)
    return (m.group(1) if m else text).strip()


def run_check(code, tests, entry_point):
    """Exec code + asserts in a subprocess. Returns (passed, reason)."""
    if BANNED.search(code):
        return False, "banned-import"
    prog = (PREAMBLE + code + "\n\n" + tests +
            f"\ncheck({entry_point})\nprint('ASSERTS-OK')\n")
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(prog)
        path = f.name
    try:
        p = subprocess.run([sys.executable, path], capture_output=True,
                           text=True, timeout=15)
        if p.returncode == 0 and "ASSERTS-OK" in p.stdout:
            return True, "pass"
        err = (p.stderr.strip().splitlines() or [""])[-1][:200]
        return False, f"assert-fail: {err}"
    except subprocess.TimeoutExpired:
        return False, "timeout"
    finally:
        os.unlink(path)


def main():
    ap = argparse.ArgumentParser(description="alfa-1.2 eval gate")
    ap.add_argument("--endpoint", default="http://127.0.0.1:8080")
    ap.add_argument("--test-file", required=True)
    ap.add_argument("--n-tasks", type=int, default=20)
    ap.add_argument("--n-predict", type=int, default=256)
    ap.add_argument("--min-pass", type=float, default=0.10)
    ap.add_argument("--out", default="reports/eval.json")
    ap.add_argument("--eval-log", default=None,
                    help="merged/eval_log.json from train_qlora.py")
    ap.add_argument("--seeds-bn", default=None,
                    help="bengali-qa.jsonl for human-graded spot checks")
    ap.add_argument("--model", default="alfa-1.2-exp1")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.test_file, encoding="utf-8")
            if l.strip()][:args.n_tasks]
    passed, failures = 0, []
    for r in rows:
        tid = r.get("task_id") or r.get("entry_point") or "?"
        try:
            gen = api_post(args.endpoint, "/completion",
                           {"prompt": r["prompt"], "n_predict": args.n_predict,
                            "temperature": 0})["content"]
            ok, reason = run_check(extract_code(gen), r.get("tests", ""),
                                   r.get("entry_point", ""))
        except Exception as e:
            ok, reason = False, f"harness-error: {type(e).__name__}: {e}"[:200]
        passed += ok
        if not ok:
            failures.append({"task_id": tid, "reason": reason})
        print(f"[{'PASS' if ok else 'FAIL'}] {tid} ({reason})", flush=True)

    p1 = passed / max(1, len(rows))
    bn_pending = []
    if args.seeds_bn and Path(args.seeds_bn).exists():
        bn = [json.loads(l) for l in open(args.seeds_bn, encoding="utf-8")
              if l.strip()][:10]
        for b in bn:
            msgs = b.get("messages", [])
            q = next((m["content"] for m in msgs if m["role"] == "user"), "")
            try:
                a = api_post(args.endpoint, "/completion",
                             {"prompt": q, "n_predict": 120,
                              "temperature": 0})["content"]
            except Exception as e:
                a = f"HARNESS-ERROR: {e}"
            bn_pending.append({"q": q[:200], "a": a[:500]})
    elif args.seeds_bn:
        print(f"[warn] seeds file missing, skipping Bangla checks: {args.seeds_bn}")

    eval_loss = None
    if args.eval_log and Path(args.eval_log).exists():
        try:
            eval_loss = json.load(open(args.eval_log)).get("eval_loss")
        except Exception:
            pass

    verdict = "PASS" if p1 >= args.min_pass else "FAIL"
    report = {"model": args.model, "n_tasks": len(rows), "passed": passed,
              "pass_at_1": round(p1, 4), "min_pass": args.min_pass,
              "failures": failures, "bangla_human_grade": bn_pending,
              "eval_loss": eval_loss, "verdict": verdict}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(report, open(args.out, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"pass@1 = {passed}/{len(rows)} = {p1:.3f} "
          f"(gate >= {args.min_pass}) -> {verdict}")
    print(f"report -> {args.out}")
    sys.exit(0 if verdict == "PASS" else 1)


if __name__ == "__main__":
    main()
