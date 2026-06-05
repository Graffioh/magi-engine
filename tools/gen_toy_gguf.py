#!/usr/bin/env python3
"""Toy-model GGUF fixture for the GGUF->Model adapter test. - by opus

Writes `tests/gguf/toy.gguf` carrying the EXACT toy weights from
tools/gen_golden_weights.py, but under their REAL ggml tensor names
(token_embd.weight, blk.{i}.attn_q.weight, ...) and llama.* metadata keys. The
C++ adapter (src/gguf_model.cpp) is therefore exercised on the genuine
llama.cpp naming/keys, while the resulting Model is still verified against the
*existing* golden.logits / golden.gen_ids -- nothing is recomputed.

Crucially we IMPORT make_weights() and the dims from gen_golden_weights.py
(same directory) so the weight bytes are byte-identical to what golden.logits
was computed from. We do NOT reseed or regenerate with different values.

Writer: the canonical gguf.GGUFWriter, so the file matches the real format our
C++ reader parses. add_tensor stores ggml ne[] reversed from the numpy shape, so
passing each weight as (out, in) makes the reader recover shape {out, in} --
exactly what every LinearLayer/EmbeddingLayer expects.

Run:  python3 tools/gen_toy_gguf.py
"""

import sys
from pathlib import Path

import gguf
import numpy as np

# Import the toy weight generator + dims from the golden harness next to us, so
# the GGUF carries byte-identical weights (same seed, same make_weights()).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from gen_golden_weights import (  # noqa: E402
    HEAD_DIM,
    HIDDEN,
    INTERMEDIATE,
    N_HEADS,
    N_KV_HEADS,
    N_LAYERS,
    VOCAB,
    make_weights,
)

ARCH = "llama"
CONTEXT_LENGTH = 16  # must be >= T + MAX_NEW_TOKENS (matches the C++ RoPE cache)
RMS_EPS = 1e-5
ROPE_FREQ_BASE = 10000.0

OUT_DIR = Path(__file__).resolve().parent.parent / "tests" / "gguf"
GGUF_PATH = OUT_DIR / "toy.gguf"


def ggml_name(name):
    """Map a gen_golden_weights.py weight name to its canonical ggml name."""
    fixed = {
        "embed.weight": "token_embd.weight",
        "final_norm.weight": "output_norm.weight",
        "lm_head.weight": "output.weight",
    }
    if name in fixed:
        return fixed[name]

    # Per-layer weights: "layer{i}.<suffix>" -> "blk.{i}.<ggml suffix>".
    prefix, _, suffix = name.partition(".")
    assert prefix.startswith("layer"), f"unexpected weight name: {name}"
    i = prefix[len("layer"):]
    layer_map = {
        "attn_norm.weight": "attn_norm.weight",
        "attn.q.weight": "attn_q.weight",
        "attn.k.weight": "attn_k.weight",
        "attn.v.weight": "attn_v.weight",
        "attn.o.weight": "attn_output.weight",
        "ffn_norm.weight": "ffn_norm.weight",
        "mlp.gate.weight": "ffn_gate.weight",
        "mlp.up.weight": "ffn_up.weight",
        "mlp.down.weight": "ffn_down.weight",
    }
    assert suffix in layer_map, f"unexpected layer weight suffix: {suffix}"
    return f"blk.{i}.{layer_map[suffix]}"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    weights = make_weights()  # byte-identical to the golden weights

    writer = gguf.GGUFWriter(str(GGUF_PATH), arch=ARCH)

    # llama.* hyperparameters the C++ config_from_gguf reads.
    writer.add_block_count(N_LAYERS)
    writer.add_embedding_length(HIDDEN)
    writer.add_feed_forward_length(INTERMEDIATE)
    writer.add_head_count(N_HEADS)
    writer.add_head_count_kv(N_KV_HEADS)
    writer.add_rope_dimension_count(HEAD_DIM)
    writer.add_context_length(CONTEXT_LENGTH)
    writer.add_layer_norm_rms_eps(RMS_EPS)  # stored float32
    writer.add_rope_freq_base(ROPE_FREQ_BASE)
    writer.add_vocab_size(VOCAB)

    # Weights under their canonical ggml names. numpy shape is (out, in); the
    # writer reverses it to ne=[in, out], and our reader un-reverses back to
    # {out, in}, so the loaded views match what the toy model expects.
    name_map = {}
    for name, arr in weights.items():
        gname = ggml_name(name)
        name_map[name] = gname
        writer.add_tensor(gname, np.ascontiguousarray(arr, dtype=np.float32))

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    # Re-read with GGUFReader and verify shapes + a couple of values round-trip.
    reader = gguf.GGUFReader(str(GGUF_PATH))
    read_tensors = {t.name: t for t in reader.tensors}
    for name, arr in weights.items():
        gname = name_map[name]
        assert gname in read_tensors, f"missing tensor in re-read: {gname}"
        rt = read_tensors[gname]
        # GGUFReader exposes ggml ne[] order (reversed); reverse back to numpy shape.
        got_shape = tuple(int(d) for d in reversed(rt.shape))
        assert got_shape == arr.shape, (
            f"shape mismatch for {gname}: got {got_shape}, want {arr.shape}"
        )
        got = np.array(rt.data, dtype=np.float32).reshape(arr.shape)
        assert np.allclose(got, arr, atol=0, rtol=0), f"value mismatch for {gname}"

    print(f"wrote {GGUF_PATH}")
    print(f"  arch={ARCH}  n_layers={N_LAYERS} hidden={HIDDEN} intermediate={INTERMEDIATE}")
    print(f"  n_heads={N_HEADS} n_kv_heads={N_KV_HEADS} head_dim={HEAD_DIM} "
          f"vocab={VOCAB} context_length={CONTEXT_LENGTH}")
    print(f"  {len(weights)} tensors round-tripped via GGUFReader (shapes + values OK)")
    for name in ("embed.weight", "lm_head.weight", "layer0.attn.q.weight"):
        print(f"  {name:22s} -> {name_map[name]:24s} "
              f"first={weights[name].flatten()[0]:.6f}")


if __name__ == "__main__":
    main()
