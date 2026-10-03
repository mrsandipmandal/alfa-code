"""Convert HF checkpoint -> GGUF via llama.cpp. Needs: git clone https://github.com/ggerganov/llama.cpp"""
import subprocess, sys
from pathlib import Path

def main(ckpt="outputs/tiny-50M", quant="Q4_K_M", out="outputs/tiny-50M.gguf"):
    conv = Path("third_party/llama.cpp/convert_hf_to_gguf.py")
    if not conv.exists():
        print("Clone llama.cpp first: git clone https://github.com/ggerganov/llama.cpp third_party/llama.cpp")
        sys.exit(1)
    subprocess.run([sys.executable, str(conv), ckpt, "--outfile", out, "--outtype", "f16"], check=True)
    quantize = Path("third_party/llama.cpp/build/bin/llama-quantize")
    if quantize.exists():
        subprocess.run([str(quantize), out, out.replace(".gguf", f".{quant}.gguf"), quant], check=True)
    print(f"GGUF ready -> {out}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="outputs/tiny-50M")
    ap.add_argument("--quant", default="Q4_K_M")
    ap.add_argument("--out", default="outputs/tiny-50M.gguf")
    a = ap.parse_args()
    main(a.ckpt, a.quant, a.out)
