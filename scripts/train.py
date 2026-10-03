"""Tiny Llama pretrain demo (CPU/4GB friendly). For 1B+ use cloud + DeepSpeed."""
import json
from pathlib import Path
import torch
from transformers import LlamaConfig, LlamaForCausalLM, Trainer, TrainingArguments
from torch.utils.data import Dataset

class JsonlDS(Dataset):
    def __init__(self, path, tok_len=512):
        self.rows = [json.loads(l) for l in open(path, encoding="utf-8")]
        self.tok_len = tok_len
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        import random
        r = self.rows[i]
        # char-level pseudo-tokenization for demo (real run uses train_tokenizer.py + proper encode)
        s = (r.get("prompt", "") + "\n" + r.get("think", "") + "\n" + r.get("answer", ""))[:2000]
        ids = [min(ord(c), 30000) for c in s][:self.tok_len]
        ids += [0] * (self.tok_len - len(ids))
        return {"input_ids": torch.tensor(ids, dtype=torch.long), "labels": torch.tensor(ids, dtype=torch.long)}

def main():
    cfg = LlamaConfig(hidden_size=512, num_hidden_layers=8, num_attention_heads=8,
                      num_key_value_heads=4, intermediate_size=1376, vocab_size=32000, max_position_embeddings=2048)
    model = LlamaForCausalLM(cfg)
    ds = JsonlDS("data/processed/train.jsonl")
    args = TrainingArguments(output_dir="outputs/tiny-50M", per_device_train_batch_size=2,
                             num_train_epochs=1, logging_steps=5, save_steps=50, fp16=torch.cuda.is_available())
    Trainer(model=model, args=args, train_dataset=ds).train()
    model.save_pretrained("outputs/tiny-50M")
    print("saved -> outputs/tiny-50M")

if __name__ == "__main__":
    main()
