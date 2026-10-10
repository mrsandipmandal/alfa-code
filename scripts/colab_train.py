"""Train code-100k on a Colab VM (single T4, seq 4096), then tar the output.

Run remotely:
  colab exec -s <session> -f scripts/colab_train.py --timeout 43200

Single T4 budget: adafactor (config default — sublinear optimizer state)
+ grad-accum 16 + grad ckpt.
Colab T4 = 16GB (a bit roomier than Kaggle's 15GB view).
"""
import os
import subprocess

WORK = "/content/alfa"
OUT_NAME = "code-100k"
SEQ = os.environ.get("ALFA_SEQ", "4096")
EPOCHS = os.environ.get("ALFA_EPOCHS", "1")
DATA = os.environ.get("ALFA_DATA", f"{WORK}/data/processed/train.jsonl")
RESUME = os.environ.get("ALFA_RESUME", "")  # set to OUT dir to continue


def sh(cmd: str):
    print(f"$ {cmd}", flush=True)
    r = subprocess.run(cmd, shell=True)
    if r.returncode != 0:
        raise SystemExit(f"FAILED ({r.returncode}): {cmd}")


os.chdir(WORK)
out = f"{WORK}/outputs/{OUT_NAME}"
resume = f" --resume-from {out}" if RESUME else ""

sh(f"nvidia-smi --query-gpu=name,memory.total --format=csv")
sh(f"df -h /content | tail -1")
sh("PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0 "
   f"python scripts/train_hf.py --data {DATA} --out {out} "
   f"--epochs {EPOCHS} --config configs/code-100k.yaml "
   f"--max-seq-len {SEQ} --batch-size 1 --grad-accum 16 "
   "--save-steps 100 --save-total-limit 1" + resume)

# one artifact for `colab download`
sh(f"ls -lh {out}")
sh(f"tar -C {WORK}/outputs -czf /content/{OUT_NAME}.tar.gz {OUT_NAME}")
sh(f"ls -lh /content/{OUT_NAME}.tar.gz")
print(f"TRAIN DONE -> /content/{OUT_NAME}.tar.gz", flush=True)
