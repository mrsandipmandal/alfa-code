"""QLoRA SFT for alfa-1.2: base-1B (4-bit NF4) + LoRA on mixed-1.2 messages.

Pack format MUST match the GGUF chat template (convert_to_gguf.py default):
    user_text + "\\n" + assistant_text + "</s>"   (no BOS prefix)
Loss is computed on ASSISTANT tokens only (prompt part masked with -100).

Stages:
  1. --dry-run   : no GPU/model needed. Builds the packed+masked batches from
                   --data, prints shapes + mask ratio + alignment rate. Use it
                   locally before burning Kaggle quota.
  2. full run    : 4-bit base -> LoRA -> train -> eval log -> save adapters ->
                   merge to full fp16 -> optional Hub push.
  3. GGUF        : scripts/convert_to_gguf.py --ckpt <merged> (template matches)

Kaggle cell (after scritps installed + data built):
  !pip install -q peft bitsandbytes accelerate
  !python scripts/train_qlora.py --config configs/alfa-1.2.yaml

Experiment first (quota-safe):
  !python scripts/train_qlora.py --config configs/alfa-1.2.yaml --max-rows 800 --epochs 1
"""
import argparse
import json
import sys
from pathlib import Path

import yaml


def load_cfg(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def pack_text(messages, sep, eos):
    user = " ".join(m["content"] for m in messages if m["role"] == "user")
    asst = " ".join(m["content"] for m in messages if m["role"] == "assistant")
    return user + sep + asst + eos, asst


def find_sub(full, sub):
    """Leftmost index of sublist sub in full, or None (fast path)."""
    if not sub or len(sub) > len(full):
        return None
    first, n, i = sub[0], len(sub), 0
    while True:
        try:
            i = full.index(first, i)
        except ValueError:
            return None
        if full[i:i + n] == sub:
            return i
        i += 1


def build_features(rows, tok, cfg, max_rows=None, check_n=200):
    """Tokenize + mask prompts. Returns (features, report)."""
    pk = cfg["pack"]
    ML = int(cfg["training"]["max_seq_len"])
    feats, checked = [], 0
    early_dropped, stripped, dropped = [0], [0], 0
    for r in (rows if max_rows is None else rows[:max_rows]):
        full_text, asst_text = pack_text(r["messages"], pk["sep"], pk["eos"])
        full = tok(full_text, add_special_tokens=False,
                   truncation=True, max_length=ML)["input_ids"]
        asst = tok(asst_text, add_special_tokens=False,
                   truncation=True, max_length=ML)["input_ids"]
        # assistant block may merge with the trailing "\n" of the prompt
        # (e.g. "\nSure" is one token) — strip up to 8 head tokens of the
        # assistant side until the tail is found verbatim in full.
        start, used = None, 0
        for drop in range(0, min(8, len(asst))):
            sub = asst[drop:]
            if not sub:
                break
            pos = find_sub(full, sub)
            if pos is not None:
                start, used = pos, drop
                break
        if checked < check_n:
            checked += 1
            if start is None:
                early_dropped[0] += 1
            elif used > 0:
                stripped[0] += 1
        if start is None:
            dropped += 1  # never train mis-masked
            continue
        labels = [-100] * start + full[start:]
        feats.append({"input_ids": full, "labels": labels})
    report = {"n": len(feats), "align_checked": checked,
              "align_dropped": early_dropped[0], "stripped": stripped[0],
              "dropped_misalign": dropped}
    return feats, report


def align_fatal(rep):
    c = rep["align_checked"]
    return c > 0 and rep["align_dropped"] / c > 0.10


def dry_run(cfg, args):
    from transformers import LlamaTokenizerFast
    tk = cfg["tokenizer"]
    tok = LlamaTokenizerFast(tokenizer_file=tk["file"], bos_token=tk.get("bos", "<s>"),
                             eos_token=tk.get("eos", "</s>"),
                             unk_token="<unk>", pad_token="<pad>")
    rows = read_jsonl(args.train or cfg["data"]["train"])
    feats, rep = build_features(rows, tok, cfg, args.max_rows)
    n_tok = sum(len(f["input_ids"]) for f in feats)
    n_lab = sum(1 for f in feats for l in f["labels"] if l != -100)
    print(f"rows={len(rows)} feats={len(feats)} tokens={n_tok} "
          f"supervised={n_lab} ({100.0 * n_lab / max(1, n_tok):.1f}%)")
    print(f"align: checked={rep['align_checked']} dropped={rep['align_dropped']} "
          f"stripped={rep['stripped']} total-dropped={rep['dropped_misalign']}")
    if feats:
        print("sample ids:", feats[0]["input_ids"][:12], "...")
        print("sample txt:", tok.decode(feats[0]["input_ids"][:60]))
    if align_fatal(rep):
        print("FATAL: prompt/answer boundary misaligns — fix pack format.",
              file=sys.stderr)
        sys.exit(3)
    print("DRY-RUN OK")


def train(cfg, args):
    import torch
    from transformers import (LlamaForCausalLM, LlamaTokenizerFast,
                              TrainingArguments, Trainer,
                              BitsAndBytesConfig, DataCollatorForSeq2Seq)
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    tr = cfg["training"]
    tk = cfg["tokenizer"]
    torch.manual_seed(tr.get("seed", 42))

    tok = LlamaTokenizerFast(tokenizer_file=tk["file"], bos_token=tk.get("bos", "<s>"),
                             eos_token=tk.get("eos", "</s>"),
                             unk_token="<unk>", pad_token="<pad>")
    tok.pad_token = "<pad>"

    bnb = BitsAndBytesConfig(load_in_4bit=True,
                             bnb_4bit_quant_type=cfg["quant"].get("quant_type", "nf4"),
                             bnb_4bit_use_double_quant=cfg["quant"].get("double_quant", True),
                             bnb_4bit_compute_dtype=getattr(torch, cfg["quant"].get("compute_dtype", "fp16")))
    base = cfg["model"].get("base", "")
    sub = cfg["model"].get("subfolder") or None
    print(f"loading base {base} subfolder={sub} (4-bit) ...", flush=True)
    model = LlamaForCausalLM.from_pretrained(base, subfolder=sub,
                                             quantization_config=bnb,
                                             device_map="auto")
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=tr.get("grad_ckpt", True))
    lc = cfg["lora"]
    model = get_peft_model(model, LoraConfig(
        r=int(lc.get("r", 64)), lora_alpha=int(lc.get("alpha", 16)),
        lora_dropout=float(lc.get("dropout", 0.05)),
        target_modules=list(lc.get("targets", ["q_proj", "k_proj", "v_proj",
                                               "o_proj", "gate_proj", "up_proj",
                                               "down_proj"])),
        modules_to_save=list(lc.get("modules_to_save", ["lm_head", "embed_tokens"])),
        task_type="CAUSAL_LM"))
    model.print_trainable_parameters()

    train_rows = read_jsonl(args.train or cfg["data"]["train"])
    valid_rows = read_jsonl(args.valid or cfg["data"]["valid"])
    train_feats, rep = build_features(train_rows, tok, cfg, args.max_rows)
    valid_feats, _ = build_dataset(valid_rows, tok, cfg)
    print(f"train feats={len(train_feats)} "
          f"dropped={rep['dropped_misalign']} stripped={rep['stripped']}")
    if align_fatal(rep):
        print("FATAL: boundary misaligns.", file=sys.stderr)
        sys.exit(3)

    out = args.out or tr.get("output_dir", "outputs/alfa-1.2")
    epochs = args.epochs or int(tr.get("epochs", 2))
    targs = TrainingArguments(
        output_dir=out, num_train_epochs=epochs,
        per_device_train_batch_size=int(tr.get("batch_size", 2)),
        gradient_accumulation_steps=int(tr.get("grad_accum", 8)),
        learning_rate=float(args.lr or tr.get("lr", 2e-4)),
        optim=tr.get("optim", "paged_adamw_8bit"),
        fp16=bool(tr.get("fp16", True)),
        gradient_checkpointing=tr.get("grad_ckpt", True),
        logging_steps=10, eval_strategy="steps",
        eval_steps=int(tr.get("eval_steps", 100)),
        save_steps=int(tr.get("save_steps", 200)),
        save_total_limit=int(tr.get("save_total_limit", 2)),
        report_to="none", seed=int(tr.get("seed", 42)),
        dataloader_num_workers=2,
    )
    collator = DataCollatorForSeq2Seq(tok, pad_to_multiple_of=8,
                                      return_tensors="pt")
    trainer = Trainer(model=model, args=targs,
                      train_dataset=train_feats, eval_dataset=valid_feats,
                      data_collator=collator)
    trainer.train()
    ev = trainer.evaluate()
    print(f"final eval: {ev}", flush=True)
    Path(out).mkdir(parents=True, exist_ok=True)
    with open(Path(out) / "eval_log.json", "w") as f:
        json.dump(ev, f)

    print("saving adapters ...", flush=True)
    model.save_pretrained(Path(out) / "adapters")
    print("merging to full fp16 ...", flush=True)
    merged = model.merge_and_unload()
    merged.save_pretrained(Path(out) / "merged")
    tok.save_pretrained(Path(out) / "merged")
    print(f"saved adapters + merged -> {out}", flush=True)

    if not args.no_push:
        try:
            from huggingface_hub import HfApi
            import os
            tok_hf = os.getenv("HF_TOKEN")
            hub = cfg.get("hub", {})
            repo = hub.get("repo")
            if repo:
                HfApi(token=tok_hf).upload_folder(
                    folder_path=out, repo_id=repo,
                    path_in_repo=hub.get("path_in_repo", "outputs/alfa-1.2"))
                print(f"pushed {out} -> {repo}", flush=True)
        except Exception as e:
            print(f"[warn] hub push failed (outputs still local): {e}")


def build_dataset(rows, tok, cfg):
    return build_features(rows, tok, cfg)


def main():
    ap = argparse.ArgumentParser(description="alfa-1.2 QLoRA SFT")
    ap.add_argument("--config", default="configs/alfa-1.2.yaml")
    ap.add_argument("--dry-run", action="store_true",
                    help="no GPU/model: build batches, print stats, exit")
    ap.add_argument("--max-rows", type=int, default=None,
                    help="cap train rows (experiment mode)")
    ap.add_argument("--train", default=None, help="override train jsonl path")
    ap.add_argument("--valid", default=None, help="override valid jsonl path")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-push", action="store_true")
    args = ap.parse_args()
    cfg = load_cfg(args.config)
    if args.dry_run:
        dry_run(cfg, args)
    else:
        train(cfg, args)


if __name__ == "__main__":
    main()
