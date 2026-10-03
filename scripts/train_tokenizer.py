"""Train BPE tokenizer on data/processed/train.jsonl -> tokenizers/alfa-32k.json"""
import json
from pathlib import Path
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

DATA = Path("data/processed/train.jsonl")
OUT = Path("tokenizers/alfa-32k.json")

def texts():
    for line in open(DATA, encoding="utf-8"):
        o = json.loads(line)
        yield o.get("prompt", "") + "\n" + o.get("answer", "")

def main(vocab_size=32000):
    assert DATA.exists(), f"missing {DATA}, run download_data.py first"
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size,
        special_tokens=["<s>", "</s>", "<unk>", "<pad>", "<think>", "<image>", "<video>", "<file>", "<code>"])
    tok.train_from_iterator(texts(), trainer=trainer, length=20000)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tok.save(str(OUT))
    print(f"saved -> {OUT}")

if __name__ == "__main__":
    main()
