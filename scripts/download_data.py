"""Download + normalize HF datasets to data/processed/train.jsonl

Supports coding + image + video modalities with high context length.

Unified row schema (jsonl):
  {modality, prompt, think, answer, images, video, context_len, source}

  - modality: code | image | video | code+reasoning | image+code | video+code
  - images: list[str] (urls / paths, not embedded bytes)
  - video: str | null (url / path)
  - context_len: estimated chars of prompt+answer (for long-context filtering)

Datasets (all public, Colab T4 friendly; each has fallbacks in SOURCES):
  code_small_test : HuggingFaceH4/CodeAlpaca_20K   (prompt/completion)
  code_long       : iamtarun/python_code_instructions_18k_alpaca (18k python rows, high context)
  code_instruct   : bigcode/self-oss-instruct (-> CodeAlpaca fallback)
  reasoning       : Open-Orca/OpenOrca
  image_to_code   : HuggingFaceM4/websight         (screenshot -> html)
  video_to_code   : AlexZigma/msr-vtt (-> friedrichor/MSR-VTT train_7k fallback)
  multimodal_mix  : code_long + image_to_code + video_to_code sampled together

High context: default keeps up to ~24k chars prompt + 24k answer
(~8k tokens each). Tune with --max-prompt-chars / --max-answer-chars.
Use --min-context to keep only long rows, --max-context to drop outliers.

Example (Colab T4):
  python scripts/download_data.py --dataset code_long --max-rows 500 --max-prompt-chars 24000
  python scripts/download_data.py --dataset image_to_code --max-rows 500
  python scripts/download_data.py --dataset video_to_code --max-rows 500
  python scripts/download_data.py --dataset multimodal_mix --max-rows 1500
"""
import argparse
import json
import random
from pathlib import Path

from datasets import load_dataset

MAP = {
    # coding — short instruct (public, no login needed)
    "code_small_test": ("HuggingFaceH4/CodeAlpaca_20K", None, None),
    # coding — long files, high context (public, streams well)
    "code_long": ("iamtarun/python_code_instructions_18k_alpaca", None, None),
    # coding — instruct mix
    "code_instruct": ("bigcode/self-oss-instruct", None, None),
    "reasoning": ("Open-Orca/OpenOrca", None, None),
    # image -> code (screenshot + html)
    "image_to_code": ("HuggingFaceM4/websight", None, None),
    # video -> code/caption (clip metadata + captions)
    "video_to_code": ("AlexZigma/msr-vtt", None, None),
    # combo sampler (handled specially, not a single HF id)
    "multimodal_mix": (None, None, None),
}

# Fallback sources tried in order when the primary MAP entry is
# gated / renamed / unreachable (e.g. bigcode/the-stack-smol is gated,
# microsoft/MSR-VTT does not exist). First success wins.
SOURCES = {
    "code_long": [
        ("iamtarun/python_code_instructions_18k_alpaca", None, None),
        ("HuggingFaceH4/CodeAlpaca_20K", None, None),
    ],
    "code_small_test": [
        ("HuggingFaceH4/CodeAlpaca_20K", None, None),
    ],
    "code_instruct": [
        ("bigcode/self-oss-instruct", None, None),
        ("HuggingFaceH4/CodeAlpaca_20K", None, None),
    ],
    "reasoning": [
        ("Open-Orca/OpenOrca", None, None),
    ],
    "image_to_code": [
        ("HuggingFaceM4/websight", None, None),
    ],
    "video_to_code": [
        ("AlexZigma/msr-vtt", None, None),
        ("friedrichor/MSR-VTT", "train_7k", None),
    ],
}

DEFAULT_OUT = Path("data/processed/train.jsonl")

# Placeholder tokens (must match tokenizer special tokens)
IMAGE_PH = "<image>"
VIDEO_PH = "<video>"
CODE_OPEN, CODE_CLOSE = "<code>", "</code>"


def _clip(s: str, n: int) -> str:
    s = s or ""
    return s if len(s) <= n else s[:n]


def normalize_code_row(prompt: str, answer: str, source: str,
                       max_prompt: int, max_answer: int):
    return {
        "modality": "code",
        "prompt": _clip(prompt, max_prompt),
        "think": "",
        "answer": _clip(answer, max_answer),
        "images": [],
        "video": None,
        "context_len": min(len(prompt or ""), max_prompt) + min(len(answer or ""), max_answer),
        "source": source,
    }


def normalize_image_row(instruction: str, html: str, image_ref: str,
                        source: str, max_prompt: int, max_answer: int,
                        image_tokens: int):
    prompt = f"{IMAGE_PH}\n{instruction or 'Build this UI as HTML/CSS:'}"
    prompt = _clip(prompt, max_prompt)
    answer = _clip(html or "", max_answer)
    return {
        "modality": "image+code",
        "prompt": prompt,
        "think": "",
        "answer": answer,
        "images": [image_ref] if image_ref else [],
        "video": None,
        "context_len": len(prompt) + len(answer) + image_tokens * 4,
        "source": source,
    }


def normalize_video_row(caption: str, video_ref: str, source: str,
                        max_prompt: int, max_answer: int,
                        video_frames: int, tokens_per_frame: int):
    frames_ph = " ".join([VIDEO_PH] * video_frames)
    prompt = _clip(f"{frames_ph}\nDescribe this clip, then write code to render/animate it: {caption or ''}",
                  max_prompt)
    answer = _clip(caption or "", max_answer)
    return {
        "modality": "video+code",
        "prompt": prompt,
        "think": "",
        "answer": answer,
        "images": [],
        "video": video_ref,
        "context_len": len(prompt) + len(answer) + video_frames * tokens_per_frame * 4,
        "source": source,
    }


def _load_rows(ds_id, subset, max_rows, streaming, seed=42):
    # codeparrot datasets still use a loading script -> needs explicit trust
    trust = ds_id.startswith("codeparrot/")
    if trust:
        print(f"note: trusting remote code for {ds_id} (loading script)")
    print(f"Loading {ds_id} (subset={subset}, streaming={streaming}) ...")
    ds = load_dataset(ds_id, subset, split="train", streaming=streaming,
                      trust_remote_code=trust)
    if streaming:
        it = iter(ds)
        rows = []
        for _ in range(max_rows):
            try:
                rows.append(next(it))
            except StopIteration:
                break
        return rows
    n = len(ds)
    k = min(max_rows, n)
    return list(ds.select(range(k)))


def _load_rows_first(name, max_rows, streaming, seed=42):
    """Try each candidate source in SOURCES[name]; first success wins."""
    last_err = None
    for ds_id, subset, _ in SOURCES.get(name, [MAP[name]]):
        try:
            rows = _load_rows(ds_id, subset, max_rows, streaming, seed)
            print(f"using source {ds_id} (subset={subset}) -> {len(rows)} raw rows")
            return rows, ds_id
        except Exception as e:
            print(f"[warn] {name} source {ds_id} failed ({type(e).__name__}: {e}), trying next")
            last_err = e
    raise RuntimeError(f"all sources failed for {name}: {last_err}")


def _code_text_row(r: dict):
    """Best-effort long code extraction (the-stack / codeparrot style rows)."""
    for k in ("content", "text", "code", "completion", "output", "solution"):
        v = r.get(k)
        if isinstance(v, str) and v.strip():
            lang = r.get("language") or r.get("lang") or ""
            path = r.get("path") or r.get("filename") or ""
            header = f"# file: {path} lang: {lang}\n" if (path or lang) else ""
            return header + v, ""
    # fallback: stringify small rows
    s = json.dumps(r, ensure_ascii=False, default=str)
    return s, ""


def build_dataset(name: str, max_rows: int, streaming: bool,
                  max_prompt: int, max_answer: int,
                  image_tokens: int, video_frames: int,
                  video_tokens_per_frame: int, seed: int):
    rows_out = []

    if name == "multimodal_mix":
        # Sample code_long + image_to_code + video_to_code evenly.
        parts = ["code_long", "image_to_code", "video_to_code"]
        per = max(1, max_rows // len(parts))
        for p in parts:
            try:
                rows_out += build_dataset(p, per, True, max_prompt, max_answer,
                                          image_tokens, video_frames,
                                          video_tokens_per_frame, seed)
            except Exception as e:
                print(f"[warn] {p} failed ({e}), skipping")
        random.Random(seed).shuffle(rows_out)
        return rows_out[:max_rows]

    ds_id, subset, _ = MAP[name]
    auto_stream = streaming or name in ("code_long", "image_to_code", "video_to_code")
    raw, used_id = _load_rows_first(name, max_rows, auto_stream, seed)
    ds_id = used_id  # record the source that actually worked

    def _caption(r: dict) -> str:
        c = (r.get("caption") or r.get("text") or r.get("sentence") or "")
        if isinstance(c, list):  # e.g. friedrichor/MSR-VTT captions list
            c = c[0] if c else ""
        return str(c)

    for i, r in enumerate(raw):
        try:
            if name == "code_small_test":
                rows_out.append(normalize_code_row(
                    f"{CODE_OPEN}\n{r.get('prompt', '')}\n{CODE_CLOSE}",
                    r.get("completion", ""), ds_id, max_prompt, max_answer))
            elif name == "code_long":
                prompt, _ = _code_text_row(r)
                if len(prompt) < 200 and (r.get("instruction") or r.get("input") or r.get("output")):
                    # alpaca-style python instruction rows (e.g. iamtarun 18k)
                    instr = (r.get("instruction") or "")
                    if r.get("input"):
                        instr += "\n" + r.get("input")
                    ans = r.get("output") or r.get("response") or instr
                    rows_out.append(normalize_code_row(
                        f"Complete and explain this code:\n{CODE_OPEN}\n{instr}\n{CODE_CLOSE}",
                        ans, ds_id, max_prompt, max_answer))
                else:
                    # long-context: keep file body as prompt, ask to complete/explain
                    rows_out.append(normalize_code_row(
                        f"Complete and explain this file:\n{CODE_OPEN}\n{prompt}\n{CODE_CLOSE}",
                        prompt, ds_id, max_prompt, max_answer))
            elif name in ("code_instruct", "reasoning"):
                prompt = r.get("instruction") or r.get("prompt") or r.get("question") or ""
                answer = r.get("output") or r.get("response") or r.get("answer") or ""
                obj = normalize_code_row(prompt, answer, ds_id, max_prompt, max_answer)
                obj["modality"] = "code+reasoning"
                rows_out.append(obj)
            elif name == "image_to_code":
                # websight: {image: PIL/deferred, text/html fields vary}
                html = (r.get("html") or r.get("code") or r.get("text")
                        or r.get("cleaned_html") or "")
                instr = (r.get("instruction") or r.get("prompt")
                         or "Build this UI as a single HTML file:")
                img_ref = str(r.get("image_url") or r.get("url") or r.get("id") or "")
                if not img_ref:
                    # PIL image has no URL — keep an index ref so images[] is non-empty
                    img_ref = f"{ds_id}#{i}"
                # don't embed PIL bytes in jsonl — keep reference only
                rows_out.append(normalize_image_row(
                    instr, str(html), img_ref, ds_id,
                    max_prompt, max_answer, image_tokens))
            elif name == "video_to_code":
                # MSR-VTT style: {caption, video_id/url/clip}
                caption = _caption(r)
                vref = str(r.get("video_url") or r.get("url")
                            or r.get("video_id") or r.get("clip_id") or r.get("id") or "")
                if not vref:
                    vref = f"{ds_id}#{i}"
                rows_out.append(normalize_video_row(
                    caption, vref, ds_id, max_prompt, max_answer,
                    video_frames, video_tokens_per_frame))
            else:
                rows_out.append(normalize_code_row(str(r), "", ds_id, max_prompt, max_answer))
        except Exception as e:
            print(f"[warn] skip row: {e}")
            continue
    return rows_out


def main():
    ap = argparse.ArgumentParser(description="Multimodal high-context data builder")
    ap.add_argument("--dataset", default="code_long", choices=list(MAP.keys()))
    ap.add_argument("--max-rows", type=int, default=500)
    ap.add_argument("--streaming", action="store_true",
                    help="force streaming (auto-on for code_long/image/video)")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--overwrite", action="store_true",
                    help="overwrite OUT instead of appending")
    ap.add_argument("--seed", type=int, default=42)
    # high-context budgets (chars; ~4 chars ~= 1 token)
    ap.add_argument("--max-prompt-chars", type=int, default=24000,
                    help="~6k tokens prompt budget (8k ctx). Raise to 48000 for 16k+ ctx")
    ap.add_argument("--max-answer-chars", type=int, default=24000)
    ap.add_argument("--min-context", type=int, default=0,
                    help="drop rows with context_len below this")
    ap.add_argument("--max-context", type=int, default=200000,
                    help="drop rows above this (outliers)")
    # multimodal token budgets (informational, also used by train.py)
    ap.add_argument("--image-tokens", type=int, default=256)
    ap.add_argument("--video-frames", type=int, default=4)
    ap.add_argument("--video-tokens-per-frame", type=int, default=64)
    args = ap.parse_args()

    rows = build_dataset(args.dataset, args.max_rows, args.streaming,
                         args.max_prompt_chars, args.max_answer_chars,
                         args.image_tokens, args.video_frames,
                         args.video_tokens_per_frame, args.seed)
    # filter by context length
    kept = [r for r in rows if args.min_context <= r["context_len"] <= args.max_context]
    print(f"kept {len(kept)}/{len(rows)} rows "
          f"(ctx {args.min_context}-{args.max_context})")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if args.overwrite else "a"
    # first write to a fresh file keeps old dupes away: suggest --overwrite
    with open(out, mode, encoding="utf-8") as f:
        for o in kept:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")

    n_img = sum(1 for r in kept if r["images"])
    n_vid = sum(1 for r in kept if r["video"])
    avg_ctx = sum(r["context_len"] for r in kept) // max(1, len(kept))
    print(f"Wrote {len(kept)} rows -> {out} "
          f"(with_images={n_img}, with_video={n_vid}, avg_context_chars={avg_ctx})")


if __name__ == "__main__":
    main()
