#!/bin/bash
# Phase-D release: merged safetensors -> GGUF Q4+Q8 -> load-test -> publish.
#   ./scripts/release_1_2.sh <merged-dir> [tag] [ctx]
# Example:
#   ./scripts/release_1_2.sh /kaggle/working/outputs/alfa-1.2/merged v1.2.0-exp1 2048
# Steps: convert -> tokenize-proof (must equal HF ids) -> gh release ->
# HF gguf/ upload. Prints the README row at the end (paste manually).
set -euo pipefail
MERGED="${1:?usage: release_1_2.sh <merged-dir> [tag] [ctx]}"
TAG="${2:-v1.2.0-exp1}"
CTX="${3:-2048}"
REPO="mrsandipmandal/alfa-code"
HFMODEL="mrsandip/Alfa-Code"
HERE="$(cd "$(dirname "$0")/.." && pwd)"
OUT="/tmp/alfa-rel"; mkdir -p "$OUT"

echo "== convert Q8_0 =="
python3 "$HERE/scripts/convert_to_gguf.py" --ckpt "$MERGED" --quant Q8_0 \
  --out "$OUT/alfa-1.2-q8.gguf" --tokenizer "$HERE/tokenizers/alfa-32k.json" \
  --ctx-size "$CTX"
echo "== convert Q4_0 =="
python3 "$HERE/scripts/convert_to_gguf.py" --ckpt "$MERGED" --quant Q4_0 \
  --out "$OUT/alfa-1.2-q4.gguf" --tokenizer "$HERE/tokenizers/alfa-32k.json" \
  --ctx-size "$CTX"

echo "== load-test (tokenize must be HF-identical) =="
for F in "$OUT"/alfa-1.2-q*.gguf; do
  P=8931; llama serve -m "$F" -c "$CTX" --port $P > "$OUT/serve.log" 2>&1 & SRV=$!
  OK=0
  for i in $(seq 1 24); do sleep 5
    curl -s http://127.0.0.1:$P/health 2>/dev/null | grep -q ok && { OK=1; break; }
  done
  [ "$OK" = 1 ] || { echo "FAIL: $F never ready"; kill $SRV 2>/dev/null; exit 1; }
  IDS=$(curl -s http://127.0.0.1:$P/tokenize -H 'Content-Type: application/json' \
    -d '{"content":"def fib(n):"}')
  echo "$F -> $IDS"
  echo "$IDS" | grep -q '\[258,3105,16,86,238\]' || \
    { echo "FAIL: $F tokenization != HF reference"; kill $SRV 2>/dev/null; exit 1; }
  kill $SRV 2>/dev/null; wait $SRV 2>/dev/null || true
done
echo "load-test: 2/2 PASS"

echo "== github release $TAG =="
gh release create "$TAG" --repo "$REPO" \
  --title "alfa-1.2 $TAG (QLoRA mixed-data SFT)" \
  --notes "Alfa 1.2 experimental: base-1B + QLoRA on mixed-1.2. Run: llama-server -m alfa-1.2-q4.gguf -c $CTX" \
  "$OUT"/alfa-1.2-q8.gguf "$OUT"/alfa-1.2-q4.gguf

echo "== huggingface upload =="
python3 - "$OUT" "$HFMODEL" <<'EOF'
import sys
from huggingface_hub import HfApi
d, repo = sys.argv[1], sys.argv[2]
api = HfApi()
for f in ["alfa-1.2-q8.gguf", "alfa-1.2-q4.gguf"]:
    api.upload_file(path_or_fileobj=f"{d}/{f}", path_in_repo=f"gguf/{f}",
                    repo_id=repo, repo_type="model")
    print("uploaded", f, flush=True)
EOF

Q8MB=$(du -m "$OUT/alfa-1.2-q8.gguf" | cut -f1)
Q4MB=$(du -m "$OUT/alfa-1.2-q4.gguf" | cut -f1)
echo "== DONE. Paste this README row: =="
echo "| alfa-1.2 (1.1B+LoRA) | [alfa-1.2-q8.gguf](https://huggingface.co/mrsandip/Alfa-Code/resolve/main/gguf/alfa-1.2-q8.gguf) (${Q8MB} MB) | [alfa-1.2-q4.gguf](https://huggingface.co/mrsandip/Alfa-Code/resolve/main/gguf/alfa-1.2-q4.gguf) (${Q4MB} MB) | $CTX |"
