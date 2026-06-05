#!/usr/bin/env python3
"""Independent NumPy reference forward for the REAL TinyLlama GGUF. - by opus

The golden harness (tools/gen_golden_weights.py) proves the C++ ops are correct
on a tiny, fixed-seed toy model. This script raises the bar: it runs the *same*
trusted NumPy ops over the *real* TinyLlama-1.1B weights (22 layers, hidden=2048)
loaded straight out of the GGUF, and emits the last-row logits for a fixed prompt.
The C++ engine (src/run_gguf.cpp -> magi_run) runs its own forward over the same
file and dumps its last-row logits; comparing the two vectors is the numerical
correctness gate for the whole engine on a production-scale checkpoint.

Why reuse gen_golden_weights' ops: those reference functions are already verified
1:1 against src/ops.cpp by the golden tests. Importing them here means the only
new surface area is (a) loading real f32 tensors from the GGUF and (b) wiring the
real ggml tensor names + real config into the existing block/forward logic. RoPE
is interleaved; attention already handles GQA (n_heads != n_kv_heads), the causal
mask, and the 1/sqrt(head_dim) scale -- nothing about the math changes at scale.

Serialization: the final (vocab,) last-row logits are written to
models/real_ref_logits.f32 as raw little-endian float32, the same format the C++
side dumps, so a downstream comparison just loads both and diffs them.

Run:  python3 tools/gen_real_reference.py
This finishes in seconds: every matmul is a NumPy/BLAS gemm.
"""

import sys
from pathlib import Path

import numpy as np
from gguf import GGUFReader

# Make the reference ops importable (they live next to this file in tools/).
sys.path.insert(0, str(Path(__file__).resolve().parent))
import gen_golden_weights as ref  # linear, rmsnorm, silu, mlp, embed, rope, attention

# ----------------------------------------------------------------------------
# Real model paths + the fixed test prompt. These exact ids must be used on the
# C++ side too (magi_run --ids 1,22172,29892,590,1024) or the comparison is moot.
# ----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "tinyllama-1.1b-chat-f32.gguf"
OUT_PATH = ROOT / "models" / "real_ref_logits.f32"

IDS = [1, 22172, 29892, 590, 1024]  # 5 valid token ids (arbitrary, but fixed)

# Real TinyLlama-1.1B-Chat config (also printed by magi_run from the GGUF kv).
CFG = {
    "n_layers": 22,
    "hidden": 2048,
    "intermediate": 5632,
    "n_heads": 32,
    "n_kv_heads": 4,
    "head_dim": 64,
    "rope_base": 10000,
    "rms_eps": np.float32(1e-5),
    "vocab": 32000,
}


# ----------------------------------------------------------------------------
# Weight loading. GGUFReader exposes each tensor's `.data` as a numpy array in
# C row-major order == reversed(ne) == LOGICAL (out, in) shape -- exactly what
# the reference `linear(x, W)` (which does x @ W.T) wants. So no transpose or
# reshape is needed: we just grab .data as float32. We only materialize the
# tensors we actually use (all of them, here) into a name->array dict.
# ----------------------------------------------------------------------------
def load_weights(path):
    reader = GGUFReader(str(path))
    w = {}
    for t in reader.tensors:
        # .data is already numpy, C-contiguous, shape == reversed(ne).
        w[t.name] = np.asarray(t.data, dtype=np.float32)
    return w


def transformer_block(x, w, i, cfg):
    """One pre-norm decoder block over real ggml-named weights for layer i.

    Mirrors gen_golden_weights.transformer_block / Model::forward exactly, but
    with the ggml naming scheme (blk.{i}.attn_norm.weight, blk.{i}.attn_q.weight,
    ...) and the real config dims. start_pos is 0 (single prefill of the prompt).
        h   = x + attention(attn_norm(x))
        out = h + mlp(ffn_norm(h))
    """
    p = f"blk.{i}."
    attn_out = ref.attention(
        ref.rmsnorm(x, w[p + "attn_norm.weight"], cfg["rms_eps"]),
        w[p + "attn_q.weight"],
        w[p + "attn_k.weight"],
        w[p + "attn_v.weight"],
        w[p + "attn_output.weight"],
        cfg["n_heads"],
        cfg["n_kv_heads"],
        cfg["head_dim"],
        cfg["rope_base"],
        start_pos=0,
    )
    h = (x + attn_out).astype(np.float32)
    mlp_out = ref.mlp(
        ref.rmsnorm(h, w[p + "ffn_norm.weight"], cfg["rms_eps"]),
        w[p + "ffn_gate.weight"],
        w[p + "ffn_up.weight"],
        w[p + "ffn_down.weight"],
    )
    return (h + mlp_out).astype(np.float32)


def real_forward(ids, w, cfg):
    """Whole-model forward -> logits (T, vocab). Mirrors Model::forward in
    src/model.cpp: embed -> 22 pre-norm blocks (residual stream) -> output_norm
    -> linear(output.weight). Untied lm_head (output.weight is present).
    """
    h = ref.embed(ids, w["token_embd.weight"])  # (T, hidden)
    for i in range(cfg["n_layers"]):
        h = transformer_block(h, w, i, cfg)
    h = ref.rmsnorm(h, w["output_norm.weight"], cfg["rms_eps"])
    return ref.linear(h, w["output.weight"])  # (T, vocab)


def main():
    print(f"loading {MODEL_PATH.name} ...")
    w = load_weights(MODEL_PATH)

    # Sanity-check a shape so a silent reversed/transposed load fails loudly.
    emb = w["token_embd.weight"]
    assert emb.shape == (CFG["vocab"], CFG["hidden"]), (
        f"token_embd.weight shape {emb.shape} != expected "
        f"({CFG['vocab']}, {CFG['hidden']}) -- GGUF load may be transposed"
    )
    print(f"  token_embd.weight shape = {emb.shape}  (OK)")

    print(f"running NumPy reference forward on ids={IDS} ...")
    logits = real_forward(IDS, w, CFG)  # (T, vocab)
    last = logits[-1].astype(np.float32)  # (vocab,)

    # Dump last-row logits as raw little-endian float32 for the C++ comparison.
    last.astype("<f4").tofile(OUT_PATH)
    print(f"wrote last-row logits ({last.shape[0]} floats) -> {OUT_PATH}")

    # Report argmax + top-5 (id, logit), descending by logit.
    argmax = int(np.argmax(last))
    top5 = np.argsort(last)[::-1][:5]
    print(f"\nargmax token id = {argmax}   logit = {last[argmax]:.6f}")
    print("top-5 (id, logit):")
    for tid in top5:
        print(f"  {int(tid):6d}  {last[int(tid)]:.6f}")


if __name__ == "__main__":
    main()
