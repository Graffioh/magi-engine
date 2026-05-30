#!/usr/bin/env python3
"""Golden reference harness for the toy model. - by opus

Generates a small, fixed-seed toy weight set and a NumPy reference forward
pass, then writes both the weights and the expected ("golden") outputs to
`tests/golden/` as raw little-endian float32 blobs. The C++ tests load the
*same* weight bytes, run the engine's ops, and compare against the goldens.

Why this exists: attention (RoPE + GQA + causal mask + softmax) is impossible
to hand-compute reliably, so we let NumPy produce the trusted answers. This
file is the single source of truth for op correctness from here on.

IMPORTANT: every reference function below must match the C++ op in src/ops.cpp
*exactly*, including float32 arithmetic. If you change an op's math in C++,
change it here too and regenerate.

Serialization: one `<name>.f32` file per tensor (raw little-endian float32,
row-major). Shapes are fixed by the toy dims below and known to the C++ test;
`manifest.txt` records them for reference. Run:  python3 tools/gen_golden_weights.py
"""

from pathlib import Path

import numpy as np

# ----------------------------------------------------------------------------
# Toy dimensions (must match docs/roadmap.md Stage 3).
# n_heads * head_dim == hidden (8); n_kv_heads=1 exercises grouped-query attn.
# ----------------------------------------------------------------------------
VOCAB = 32
HIDDEN = 8
INTERMEDIATE = 16
N_LAYERS = 2
N_HEADS = 2
N_KV_HEADS = 1
HEAD_DIM = 4
RMS_EPS = np.float32(1e-5)  # matches the C++ default

SEED = 1234
# Fixed input token ids for the golden forward pass (T=3). The C++ test uses
# these exact ids. All are < VOCAB.
INPUT_IDS = [5, 0, 11]

OUT_DIR = Path(__file__).resolve().parent.parent / "tests" / "golden"


# ----------------------------------------------------------------------------
# Reference ops. Each mirrors src/ops.cpp 1:1, in float32.
# ----------------------------------------------------------------------------
def linear(x, W):
    """y = x @ W^T. W is stored (out_features, in_features), like LinearLayer."""
    return (x @ W.T).astype(np.float32)


def rmsnorm(x, w, eps=RMS_EPS):
    """rms = sqrt(mean(x^2) + eps); out = x/rms * w  (per row, over last dim)."""
    ms = np.mean(x * x, axis=-1, keepdims=True).astype(np.float32)
    inv_rms = (np.float32(1.0) / np.sqrt(ms + eps)).astype(np.float32)
    return (x * inv_rms * w).astype(np.float32)


def silu(x):
    """x / (1 + e^-x)."""
    return (x / (np.float32(1.0) + np.exp(-x))).astype(np.float32)


def mlp(x, w_gate, w_up, w_down):
    """SwiGLU: down(silu(gate(x)) * up(x))."""
    gate = silu(linear(x, w_gate))
    up = linear(x, w_up)
    return linear(gate * up, w_down)


def embed(ids, weight):
    """Gather rows of the embedding matrix by token id."""
    return weight[np.asarray(ids, dtype=np.int64)].astype(np.float32)


def rope(x, n_heads, head_dim, base=10000, start_pos=0):
    """Interleaved RoPE, float32 throughout. Mirrors apply_rope in src/ops.cpp.

    x is (seq_len, n_heads*head_dim), row-major. For token row s, head h, and
    pair p in [0, head_dim/2):
        col_a = h*head_dim + 2*p ; col_b = col_a + 1
        pos   = start_pos + s
        freq  = base ** (-2*p / head_dim)
        theta = pos * freq
        a' = a*cos(theta) - b*sin(theta) ; b' = a*sin(theta) + b*cos(theta)
    Everything is cast to float32 so transcendentals match C libm as closely as
    possible (a few-ULP drift remains; see the C++ golden tolerance).
    """
    out = x.astype(np.float32).copy()
    seq_len = x.shape[0]
    for s in range(seq_len):
        pos = np.float32(start_pos + s)
        for h in range(n_heads):
            head_base = h * head_dim
            for p in range(head_dim // 2):
                freq = np.float32(base) ** np.float32(-2.0 * p / head_dim)
                theta = np.float32(pos * freq)
                cos_t = np.float32(np.cos(theta))
                sin_t = np.float32(np.sin(theta))
                a = out[s, head_base + 2 * p]
                b = out[s, head_base + 2 * p + 1]
                out[s, head_base + 2 * p] = np.float32(a * cos_t - b * sin_t)
                out[s, head_base + 2 * p + 1] = np.float32(a * sin_t + b * cos_t)
    return out


# ----------------------------------------------------------------------------
# Weight generation. One fixed-seed Generator -> fully reproducible.
# ----------------------------------------------------------------------------
def make_weights():
    rng = np.random.default_rng(SEED)

    def randn(*shape):
        # small values so activations stay in a sane range; float32 to match C++
        return (rng.standard_normal(shape) * 0.5).astype(np.float32)

    w = {}
    w["embed.weight"] = randn(VOCAB, HIDDEN)
    w["final_norm.weight"] = randn(HIDDEN)
    w["lm_head.weight"] = randn(VOCAB, HIDDEN)  # untied from embed (TinyLlama-style)

    qd = N_HEADS * HEAD_DIM  # 8
    kvd = N_KV_HEADS * HEAD_DIM  # 4
    for i in range(N_LAYERS):
        p = f"layer{i}."
        w[p + "attn_norm.weight"] = randn(HIDDEN)
        w[p + "attn.q.weight"] = randn(qd, HIDDEN)
        w[p + "attn.k.weight"] = randn(kvd, HIDDEN)
        w[p + "attn.v.weight"] = randn(kvd, HIDDEN)
        w[p + "attn.o.weight"] = randn(HIDDEN, qd)
        w[p + "ffn_norm.weight"] = randn(HIDDEN)
        w[p + "mlp.gate.weight"] = randn(INTERMEDIATE, HIDDEN)
        w[p + "mlp.up.weight"] = randn(INTERMEDIATE, HIDDEN)
        w[p + "mlp.down.weight"] = randn(HIDDEN, INTERMEDIATE)
    return w


# ----------------------------------------------------------------------------
# Reference forward pass -> goldens. Only covers ops that exist in C++ today
# (embed, RMSNorm, MLP). Add RoPE / attention / full-model goldens here as the
# C++ ops land -- that is the whole point of the harness.
# ----------------------------------------------------------------------------
def make_goldens(w):
    g = {}

    # embedding lookup
    x = embed(INPUT_IDS, w["embed.weight"])  # (T, HIDDEN)
    g["golden.embed_out"] = x

    # RMSNorm of the embeddings, using layer 0's attn_norm weight
    g["golden.rmsnorm_out"] = rmsnorm(x, w["layer0.attn_norm.weight"])

    # SwiGLU MLP applied directly to the embeddings, layer 0 weights
    g["golden.mlp_out"] = mlp(
        x,
        w["layer0.mlp.gate.weight"],
        w["layer0.mlp.up.weight"],
        w["layer0.mlp.down.weight"],
    )

    # RoPE vs golden. Q/K are seeded directly (small randn-style values) and
    # saved as goldens too, so the C++ RoPE unit test loads the *same* input
    # bytes -- decoupling it from matmul/rmsnorm. Q has N_HEADS heads, K has
    # N_KV_HEADS heads (GQA); both use HEAD_DIM and start_pos=0.
    rope_rng = np.random.default_rng(SEED + 1)
    seq_len = len(INPUT_IDS)  # T = 3
    q_in = (rope_rng.standard_normal((seq_len, N_HEADS * HEAD_DIM)) * 0.5).astype(np.float32)
    k_in = (rope_rng.standard_normal((seq_len, N_KV_HEADS * HEAD_DIM)) * 0.5).astype(np.float32)
    g["golden.rope_q_in"] = q_in
    g["golden.rope_k_in"] = k_in
    g["golden.rope_q_out"] = rope(q_in, N_HEADS, HEAD_DIM)
    g["golden.rope_k_out"] = rope(k_in, N_KV_HEADS, HEAD_DIM)

    # TODO (Stage 4): g["golden.attn_out"] once attention exists in C++.
    # TODO (Stage 5): g["golden.logits"].
    return g


# ----------------------------------------------------------------------------
# Serialization: raw little-endian float32, row-major. Read back to self-check.
# ----------------------------------------------------------------------------
def write_blob(path, arr):
    arr.astype("<f4").tofile(path)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    weights = make_weights()
    goldens = make_goldens(weights)
    everything = {**weights, **goldens}

    manifest_lines = [f"# input_ids = {INPUT_IDS}", "# name  shape..."]
    for name, arr in everything.items():
        write_blob(OUT_DIR / f"{name}.f32", arr)
        manifest_lines.append(f"{name} " + " ".join(str(d) for d in arr.shape))

    (OUT_DIR / "manifest.txt").write_text("\n".join(manifest_lines) + "\n")

    # Round-trip self-check: bytes on disk must decode back to the exact arrays.
    for name, arr in everything.items():
        back = np.fromfile(OUT_DIR / f"{name}.f32", dtype="<f4").reshape(arr.shape)
        assert np.array_equal(back, arr.astype(np.float32)), (
            f"round-trip mismatch: {name}"
        )

    print(f"wrote {len(everything)} blobs to {OUT_DIR}")
    print(f"  {len(weights)} weights, {len(goldens)} goldens")
    print(f"  input_ids = {INPUT_IDS}  (T={len(INPUT_IDS)})")
    for name in goldens:
        arr = goldens[name]
        print(f"  {name:24s} shape={arr.shape}  first={arr.flatten()[0]:.6f}")


if __name__ == "__main__":
    main()
