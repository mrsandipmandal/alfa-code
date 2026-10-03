"""Download + normalize HF datasets to data/processed/train.jsonl"""
import argparse, json, os
from pathlib import Path
from datasets import load_dataset

MAP = {
    "code_small_test": ("HuggingFaceH4/CodeAlpaca_20K", None, None),  # public, no login needed
    "code_pretrain": ("bigcode/the-stack-v2", None, None),  # gated, needs HF_TOKEN
    "code_instruct": ("bigcode/self-oss-instruct", None, None),
    "reasoning": ("Open-Orca/OpenOrca", None, None),
    "image_to_code": ("HuggingFaceM4/websight", None, None),
}

OUT = Path("data/processed/train.jsonl")
OUT.parent.mkdir(parents=True, exist_ok=True)

def normalize_code_instruct(row):
    # self-oss-instruct fields vary; keep raw + unified prompt
    prompt = row.get("instruction") or row.get("prompt") or row.get("question") or ""
    answer = row.get("output") or row.get("response") or row.get("answer") or ""
    return {"modality": "code+reasoning", "prompt": prompt, "think": "", "answer": answer}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="code_small_test", choices=list(MAP.keys()))
    ap.add_argument("--max-rows", type=int, default=500)
    ap.add_argument("--streaming", action="store_true", help="stream for large sets like the-stack-v2")
    args = ap.parse_args()

    ds_id, subset, text_col = MAP[args.dataset]
    print(f"Loading {ds_id} ...")
    ds = load_dataset(ds_id, subset, split="train", streaming=args.streaming)
    if args.streaming:
        it = iter(ds)
        rows = [next(it) for _ in range(args.max_rows)]
    else:
        ds = ds.select(range(min(args.max_rows, len(ds))))
        rows = list(ds)

    with open(OUT, "a", encoding="utf-8") as f:
        for r in rows:
            if args.dataset == "code_small_test":
                # CodeAlpaca_20K: {prompt, completion}
                obj = {"modality": "code+reasoning", "prompt": r.get("prompt", "")[:4000], "think": "", "answer": r.get("completion", "")[:8000]}
            elif args.dataset in ("code_instruct", "reasoning"):
                obj = normalize_code_instruct(r)
            else:
                obj = {"modality": "image", "prompt": str(r)[:4000], "think": "", "answer": ""}
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows)} rows -> {OUT}")

if __name__ == "__main__":
    main()
