"""Convert HF Llama checkpoint -> GGUF, self-contained (no llama.cpp needed).

Uses pip `gguf` for writing + optional pure-python quantization.
Tensor map covers standard Llama (incl. GQA kv-heads). Tokenizer comes
from tokenizers/alfa-32k.json (byte-level BPE -> gguf gpt2 type).

RTX 3050 4GB: Q8_0 tiny-50M ~= 55MB, Q4_0 ~= 32MB — both trivial to run
(llama.cpp server, llama-cpp-python, or Ollama Modelfile).

  python scripts/convert_to_gguf.py --ckpt outputs/tiny-50M --quant Q8_0
  python scripts/convert_to_gguf.py --ckpt outputs/tiny-50M --quant Q4_0 --out outputs/tiny-50M-q4.gguf
  python scripts/convert_to_gguf.py --ckpt outputs/tiny-50M --quant F16
"""
import argparse
import json
from pathlib import Path

import numpy as np

QUANT_CHOICES = ("F16", "Q8_0", "Q4_0")
# NOTE: K-quants (Q4_K_M etc.) are NOT offered: our intermediate_size=1376
# is not a multiple of the K-quant block size (256), and they need the
# llama.cpp `llama-quantize` binary. Q8_0/Q4_0 use block 32 (1376 = 43*32).


def bytes_to_unicode():
    """Standard GPT-2 byte mapping: char -> byte value."""
    bs = (list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1))
          + list(range(ord("®"), ord("ÿ") + 1)))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for c, b in zip(cs, bs)}


def load_hf_weights(ckpt: Path) -> dict:
    sd = ckpt / "model.safetensors"
    if sd.exists():
        from safetensors.torch import load_file
        print(f"loading {sd}", flush=True)
        return {k: v.float().numpy() for k, v in load_file(str(sd)).items()}
    bf = ckpt / "pytorch_model.bin"
    if bf.exists():
        import torch
        print(f"loading {bf}", flush=True)
        return {k: v.float().numpy() for k, v in torch.load(str(bf), map_location="cpu").items()}
    raise SystemExit(f"no weights found in {ckpt} (need model.safetensors or pytorch_model.bin)")


def _write_yarn(writer, factor: float, original: int, rs: dict):
    """Long-context (YaRN) rope metadata — what llama.cpp reads for -c > trained.

    gguf-py versions differ in which keys they expose, so each writer call
    is probed: missing keys just mean llama.cpp falls back to its own
    dynamic rope scaling when the requested -c exceeds trained ctx.
    """
    pairs = [
        ("add_rope_scaling_type", str(rs.get("type", "yarn"))),
        ("add_rope_scale_factor", float(factor)),
        ("add_rope_scale_orig_ctx", int(original)),
        ("add_yarn_attn_factor", float(rs.get("attention_factor", 1.0))),
        ("add_yarn_beta_fast", float(rs.get("beta_fast", 32.0))),
        ("add_yarn_beta_slow", float(rs.get("beta_slow", 1.0))),
    ]
    written, skipped = [], []
    for meth, val in pairs:
        if hasattr(writer, meth):
            getattr(writer, meth)(val)
            written.append(meth)
        else:
            skipped.append(meth)
    print(f"yarn metadata: wrote {written or 'none'}"
          + (f" (gguf-py lacks: {', '.join(skipped)})" if skipped else ""), flush=True)


def main(ckpt: str, quant: str, out: str, tokenizer: str, ctx_size: int | None = None):
    import gguf

    ckpt_p, out_p = Path(ckpt), Path(out)
    cfg = json.loads((ckpt_p / "config.json").read_text())
    arch = cfg.get("architectures", ["LlamaForCausalLM"])[0]
    if "lama" not in arch.lower() and "llama" not in arch.lower():
        print(f"WARNING: arch {arch} is not Llama — tensor map may not fit", flush=True)

    hidden = cfg["hidden_size"]
    heads = cfg["num_attention_heads"]
    kv_heads = cfg.get("num_key_value_heads", heads)
    layers = cfg["num_hidden_layers"]
    ffn = cfg["intermediate_size"]
    head_dim = hidden // heads

    state = load_hf_weights(ckpt_p)

    def w(*names):
        for n in names:
            if n in state:
                return state[n]
        raise KeyError(f"none of {names} in checkpoint")

    qtype = {"F16": gguf.GGMLQuantizationType.F16,
             "Q8_0": gguf.GGMLQuantizationType.Q8_0,
             "Q4_0": gguf.GGMLQuantizationType.Q4_0}[quant]
    # norms stay F32 (llama.cpp convention)
    f32 = gguf.GGMLQuantizationType.F32

    writer = gguf.GGUFWriter(str(out_p), "llama")  # arch preset by constructor
    trained_ctx = int(cfg.get("max_position_embeddings", 2048))
    rs = cfg.get("rope_scaling") or {}
    # target context: --ctx-size wins, else rope_scaling in config, else trained
    target_ctx = (ctx_size or
                  int(float(rs.get("factor", 1.0)) *
                      int(rs.get("original_max_position_embeddings", trained_ctx))) or
                  trained_ctx)
    writer.add_context_length(target_ctx)
    writer.add_embedding_length(hidden)
    writer.add_block_count(layers)
    writer.add_feed_forward_length(ffn)
    writer.add_head_count(heads)
    writer.add_head_count_kv(kv_heads)
    writer.add_rope_dimension_count(head_dim)
    writer.add_rope_freq_base(cfg.get("rope_theta", 10000.0))
    if target_ctx > trained_ctx:
        _write_yarn(writer, target_ctx / trained_ctx, trained_ctx, rs)
        print(f"long context: trained @ {trained_ctx} -> target {target_ctx} "
              f"(llama.cpp -c {target_ctx})", flush=True)
    writer.add_layer_norm_rms_eps(cfg.get("rms_norm_eps", 1e-6))

    # ---- tokenizer: HF byte-level BPE -> gguf gpt2 type ----
    tok_js = json.loads(Path(tokenizer).read_text())
    b2u = bytes_to_unicode()
    vocab = tok_js["model"]["vocab"]
    ordered = sorted(vocab.items(), key=lambda kv: kv[1])
    token_bytes = [bytes(b2u[ch] for ch in text) for text, _ in ordered]
    merges = tok_js["model"].get("merges", [])
    added = {t["content"]: t["id"] for t in tok_js.get("added_tokens", [])}
    special = set(added)
    bos, eos = tok_js.get("bos_token"), tok_js.get("eos_token")
    unk = next((t["content"] for t in tok_js.get("added_tokens", [])
                if "unk" in t["content"].lower()), "<unk>")
    pad = next((t["content"] for t in tok_js.get("added_tokens", [])
                if "pad" in t["content"].lower()), "<pad>")
    n_vocab = len(ordered)
    types = []
    for text, _ in ordered:
        if text == unk:
            types.append(gguf.TokenType.UNKNOWN)
        elif text in special:
            types.append(gguf.TokenType.CONTROL)
        else:
            types.append(gguf.TokenType.NORMAL)
    scores = [0.0] * n_vocab
    writer.add_tokenizer_model("gpt2")
    writer.add_token_list(token_bytes)
    writer.add_token_scores(scores)
    writer.add_token_types(types)
    writer.add_token_merges(merges)
    writer.add_bos_token_id(vocab.get(bos, 1) if isinstance(bos, str) else 1)
    writer.add_eos_token_id(vocab.get(eos, 2) if isinstance(eos, str) else 2)
    writer.add_unk_token_id(vocab.get(unk, 0))
    writer.add_pad_token_id(vocab.get(pad, 0))
    writer.add_add_bos_token(False)  # training rows had no BOS prefix
    writer.add_add_eos_token(False)

    # ---- tensors (gguf-py stores arrays as-is: pre-quantize here) ----
    def add(name, arr, q=qtype):
        arr = np.asarray(arr, dtype=np.float32)
        if q == gguf.GGMLQuantizationType.F32:
            writer.add_tensor(name, arr)
        elif q == gguf.GGMLQuantizationType.F16:
            writer.add_tensor(name, arr.astype(np.float16))
        else:
            writer.add_tensor(name, gguf.quantize(arr, q), raw_dtype=q)

    add("token_embd.weight", w("model.embed_tokens.weight"))
    for i in range(layers):
        p = f"model.layers.{i}."
        add(f"blk.{i}.attn_norm.weight", w(p + "input_layernorm.weight"), f32)
        add(f"blk.{i}.attn_q.weight", w(p + "self_attn.q_proj.weight"))
        add(f"blk.{i}.attn_k.weight", w(p + "self_attn.k_proj.weight"))
        add(f"blk.{i}.attn_v.weight", w(p + "self_attn.v_proj.weight"))
        add(f"blk.{i}.attn_output.weight", w(p + "self_attn.o_proj.weight"))
        add(f"blk.{i}.ffn_norm.weight", w(p + "post_attention_layernorm.weight"), f32)
        add(f"blk.{i}.ffn_gate.weight", w(p + "mlp.gate_proj.weight"))
        add(f"blk.{i}.ffn_up.weight", w(p + "mlp.up_proj.weight"))
        add(f"blk.{i}.ffn_down.weight", w(p + "mlp.down_proj.weight"))
    add("output_norm.weight", w("model.norm.weight"), f32)
    if "lm_head.weight" in state and not cfg.get("tie_word_embeddings", False):
        add("output.weight", w("lm_head.weight"))

    out_p.parent.mkdir(parents=True, exist_ok=True)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"GGUF ready -> {out_p} ({out_p.stat().st_size / 1e6:.1f} MB, {quant})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="HF Llama checkpoint -> GGUF")
    ap.add_argument("--ckpt", default="outputs/tiny-50M")
    ap.add_argument("--quant", default="Q8_0", choices=list(QUANT_CHOICES))
    ap.add_argument("--out", default="outputs/tiny-50M.gguf")
    ap.add_argument("--tokenizer", default="tokenizers/alfa-32k.json")
    ap.add_argument("--ctx-size", type=int, default=None,
                    help="deploy context written into the GGUF (e.g. 100000). "
                         "Default: rope_scaling from config.json, else trained ctx. "
                         "Run with llama.cpp -c <ctx-size> to use it.")
    a = ap.parse_args()
    main(a.ckpt, a.quant, a.out, a.tokenizer, a.ctx_size)
