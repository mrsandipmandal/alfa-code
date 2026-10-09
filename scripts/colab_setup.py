"""Bootstrap a fresh Colab VM: clone repo, install deps, build data + tokenizer.

Run remotely:
  colab upload scripts/colab_setup.py /content/colab_setup.py
  colab exec -s <session> -f scripts/colab_setup.py --timeout 1800
"""
import os
import subprocess

REPO = "https://github.com/mrsandipmandal/alfa-code.git"
WORK = "/content/alfa"
ROWS = os.environ.get("ALFA_ROWS", "5000")


def sh(cmd: str):
    print(f"$ {cmd}", flush=True)
    r = subprocess.run(cmd, shell=True)
    if r.returncode != 0:
        raise SystemExit(f"FAILED ({r.returncode}): {cmd}")


if not os.path.isdir(WORK):
    sh(f"git clone --depth 1 {REPO} {WORK}")
else:
    sh(f"git -C {WORK} fetch origin && git -C {WORK} reset --hard origin/main")
os.chdir(WORK)

sh("pip install -q transformers datasets tokenizers accelerate "
   "huggingface_hub pyyaml bitsandbytes")

# code_mix (6 code datasets incl. apps/TACO reasoning) + BPE tokenizer
sh(f"python scripts/download_data.py --dataset code_mix "
   f"--max-rows {ROWS} --overwrite")
sh("python scripts/train_tokenizer.py")
sh("ls -lh tokenizers/ data/processed/")
print("SETUP DONE", flush=True)
