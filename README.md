---
license: apache-2.0
language:
- en
pipeline_tag: text-generation
library_name: transformers
tags:
- code
- code-generation
- multimodal
- gguf
- llama
datasets:
- HuggingFaceH4/CodeAlpaca_20K
metrics:
- code_eval
- perplexity
version: 0.1.0-tiny-50M
---

# Alfa Code - Multimodal Coding GGUF

RTX 3050 4GB: use tiny-50M demo. 1B+ needs cloud 8xH100.

## 1. Data
`python scripts/download_data.py --dataset code_small_test --max-rows 500`
Output: `data/processed/train.jsonl` {modality,prompt,think,answer}

## 2. Tokenizer + Train
`python scripts/train_tokenizer.py` -> `tokenizers/alfa-32k.json` (verified 47KB)
`pip install torch --index-url https://download.pytorch.org/whl/cu121`
`python scripts/train.py` -> `outputs/tiny-50M`

Scale-up (Colab T4, ~2-4h): 5000+ rows, chat tuning, 3-5 epochs:
`python scripts/download_data.py --dataset multimodal_mix --max-rows 5000 --overwrite`
`python scripts/download_data.py --dataset chat_qa --max-rows 2000` (appends Q&A rows)
`python scripts/train_hf.py --data data/processed/train.jsonl --out /content/outputs/tiny-50M --epochs 3 --config configs/tiny-50M.yaml --max-seq-len 2048 --batch-size 1`

1B model (16GB+ GPU only, not RTX 3050 4GB): same commands with `--config configs/base-1B.yaml --out /content/outputs/base-1B` (1.12B params, ~2.2GB fp16).

## 3. GGUF (no llama.cpp clone needed)
`pip install gguf safetensors torch transformers`
`python scripts/convert_to_gguf.py --ckpt outputs/tiny-50M --quant Q8_0` (Q8_0 ~55MB, Q4_0 ~32MB, F16 ~105MB)
K-quants (Q4_K_M) need standard dims + `llama-quantize` binary — not offered (intermediate 1376).

Run local (RTX 3050 4GB):
`llama-server -m outputs/tiny-50M.gguf -c 2048` or Ollama Modelfile (`FROM ./outputs/tiny-50M.gguf`).

## 4. UI (reasoning live->collapsed + image/file/video)
`python app/ui.py` -> http://127.0.0.1:7860

## 5. Hub sync
`hf upload mrsandip/Alfa-Code . . --exclude __pycache__ --exclude .venv --commit-message "Update"`
