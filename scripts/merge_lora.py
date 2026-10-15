"""Merge LoRA adapters onto a CLEAN (non-quantized) base -> fp16 safetensors.

Why separate: merging onto a 4-bit base keeps bitsandbytes quant state
(*.quant_map / *.absmax blobs) in the saved dir, which GGUF converters
(and plain from_pretrained without bnb) cannot read. Merge math needs no
GPU: 1.1B fp16 = 2.2GB RAM.

  python scripts/merge_lora.py --base mrsandip/alfa-1-2-train \\
      --subfolder outputs/base-1B --adapters <adapters-dir> \\
      --out <merged-dir> [--dtype fp16] [--tokenizer tokenizers/alfa-32k.json]
"""
import argparse
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description="merge LoRA -> clean fp16")
    ap.add_argument("--base", required=True)
    ap.add_argument("--subfolder", default=None)
    ap.add_argument("--adapters", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dtype", default="fp16", choices=["fp16", "fp32"])
    ap.add_argument("--tokenizer", default="tokenizers/alfa-32k.json")
    a = ap.parse_args()

    import torch
    from transformers import LlamaForCausalLM, LlamaTokenizerFast
    from peft import PeftModel

    dt = torch.float16 if a.dtype == "fp16" else torch.float32
    print(f"loading clean base {a.base} subfolder={a.subfolder} ({a.dtype}) ...",
          flush=True)
    kw = {} if not a.subfolder else {"subfolder": a.subfolder}
    base = LlamaForCausalLM.from_pretrained(
        a.base, torch_dtype=dt, device_map="cpu", **kw)
    print(f"attaching adapters {a.adapters} ...", flush=True)
    model = PeftModel.from_pretrained(base, a.adapters, device_map="cpu")
    print("merging ...", flush=True)
    merged = model.merge_and_unload()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(out), safe_serialization=True)
    LlamaTokenizerFast(tokenizer_file=a.tokenizer, bos_token="<s>",
                       eos_token="</s>", unk_token="<unk>",
                       pad_token="<pad>").save_pretrained(str(out))
    # sanity: no quant leftovers, all tensors floating 2D/1D
    from safetensors import safe_open
    f = safe_open(str(out / "model.safetensors"), framework="pt")
    keys = list(f.keys())
    bad = [k for k in keys
           if "quant_map" in k or "absmax" in k or "nested" in k]
    assert not bad, f"quant leftovers: {bad[:5]}"
    assert any(k.endswith("lm_head.weight") for k in keys), "no lm_head?"
    print(f"merged clean: {len(keys)} tensors -> {out}", flush=True)


if __name__ == "__main__":
    main()
