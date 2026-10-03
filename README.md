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

## 3. GGUF
`git clone https://github.com/ggerganov/llama.cpp third_party/llama.cpp`
`python scripts/convert_to_gguf.py --ckpt outputs/tiny-50M --quant Q4_K_M --out outputs/tiny-50M.gguf`

## 4. UI (reasoning live->collapsed + image/file/video)
`python app/ui.py` -> http://127.0.0.1:7860

## 5. Hub sync
`hf upload mrsandip/Alfa-Code . . --exclude __pycache__ --exclude .venv --commit-message "Update"`
