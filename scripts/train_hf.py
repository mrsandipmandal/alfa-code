"""HF Jobs compatible tiny-train: reads /data, writes /data/outputs + Hub."""
import argparse, json, os
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.getenv("ALFA_DATA", "/data/processed/train.jsonl"))
    ap.add_argument("--out", default=os.getenv("ALFA_OUT", "/data/outputs/tiny-50M"))
    ap.add_argument("--epochs", type=int, default=1)
    args = ap.parse_args()

    import torch
    from transformers import LlamaConfig, LlamaForCausalLM, Trainer, TrainingArguments
    from torch.utils.data import Dataset

    class JsonlDS(Dataset):
        def __init__(self, path, tok_len=512):
            rows = [json.loads(l) for l in open(path, encoding="utf-8")]
            self.rows, self.tok_len = rows, tok_len
        def __len__(self): return len(self.rows)
        def __getitem__(self, i):
            r = self.rows[i]
            s = (r.get("prompt", "") + "\n" + r.get("answer", ""))[:2000]
            ids = [min(ord(c), 30000) for c in s][:self.tok_len]
            ids += [0] * (self.tok_len - len(ids))
            return {"input_ids": torch.tensor(ids, dtype=torch.long),
                    "labels": torch.tensor(ids, dtype=torch.long)}

    print(f"data: {args.data} exists={Path(args.data).exists()}")
    cfg = LlamaConfig(hidden_size=512, num_hidden_layers=8, num_attention_heads=8,
                      num_key_value_heads=4, intermediate_size=1376,
                      vocab_size=32000, max_position_embeddings=2048)
    model = LlamaForCausalLM(cfg)
    ds = JsonlDS(args.data)
    targs = TrainingArguments(output_dir=args.out, per_device_train_batch_size=2,
                              num_train_epochs=args.epochs, logging_steps=2,
                              save_steps=50, save_total_limit=1,
                              fp16=torch.cuda.is_available(), report_to="none")
    Trainer(model=model, args=targs, train_dataset=ds).train()
    model.save_pretrained(args.out)
    print(f"saved -> {args.out}")

if __name__ == "__main__":
    main()
