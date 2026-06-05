#!/usr/bin/env python3
"""Produce a plain-F32 TinyLlama GGUF for the magi-engine loader. - by opus

The C++ loader in magi-engine only understands F32 tensors. Real model files
ship quantized (Q4/Q8/...) to save space, so we need a one-time conversion that
turns a correctly-converted *quantized* Llama GGUF into a byte-for-byte
equivalent F32 GGUF, keeping ALL metadata (architecture hyperparameters AND the
full tokenizer) intact.

Source: TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF / tinyllama-1.1b-chat-v1.0.Q8_0.gguf
  Q8_0 is near-lossless and is already a *converted* Llama GGUF: the RoPE weight
  permutation, the tokenizer, and every `llama.*` hyperparameter are baked in.
  We therefore do NOT re-permute or recompute anything — we only dequantize the
  raw tensor values to float32 and copy the key/value metadata verbatim.

Why this exists: it lets the engine load a genuine 1.1B chat model (real
tokenizer, real weights) without having to implement quant decoders in C++.

How it works:
  1. Download the Q8_0 file with huggingface_hub.hf_hub_download into models/.
  2. Read it with gguf.GGUFReader.
  3. Open a gguf.GGUFWriter for models/tinyllama-1.1b-chat-f32.gguf (arch="llama").
  4. Copy every KV field faithfully — scalars, strings, AND arrays (the
     ~32000-entry tokenizer.ggml.tokens string array, the float scores array,
     and the int token_type array all survive). The copy logic mirrors the gguf
     package's own scripts/gguf_new_metadata.py: iterate reader.fields, skip the
     virtual `GGUF.*` header fields and `general.architecture` (the writer's
     ctor regenerates that), and re-add each field with its original value type
     and (for arrays) sub-type.
  5. Dequantize every tensor to float32 and add it with writer.add_tensor.
  6. Write header + kv + tensors.
  7. Self-verify the output against the source (tensor count, vocab length, the
     key llama.* scalars, and a value/shape spot-check of 3 tensors).

Tensor shape/layout note (verified against this gguf install):
  GGUFReader stores ReaderTensor.shape as the raw on-disk dims, which are in
  *ggml* order (ne[0]=fastest). But ReaderTensor.data is reshaped to
  reversed(dims) — i.e. numpy/C row-major order. gguf.quants.dequantize keeps
  that numpy shape. GGUFWriter.add_tensor takes the numpy-order .shape and
  reverses it again when serializing tensor info (write_ti_data_to_file packs
  ti.shape[n_dims-1-j]). So feeding the dequantized numpy array straight to
  add_tensor reproduces the EXACT source ne[]. We assert this in the self-check
  rather than trusting it blindly.

Run:  python3 tools/make_f32_gguf.py
  The download is ~1.17 GB and the F32 output is ~4.3-4.5 GB, so expect this to
  take several minutes and a few GB of disk.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

import gguf
from gguf import GGMLQuantizationType, GGUFReader, GGUFValueType, GGUFWriter
from huggingface_hub import hf_hub_download

# ----------------------------------------------------------------------------
# Paths / source coordinates.
# ----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
REPO_ID = "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF"
SRC_FILENAME = "tinyllama-1.1b-chat-v1.0.Q8_0.gguf"
OUT_PATH = MODELS_DIR / "tinyllama-1.1b-chat-f32.gguf"
ARCH = "llama"

# llama.* scalar keys the self-check requires to match the source exactly.
HYPERPARAM_KEYS = [
    "llama.block_count",
    "llama.embedding_length",
    "llama.attention.head_count",
    "llama.attention.head_count_kv",
    "llama.context_length",
    "llama.rope.freq_base",
    "llama.attention.layer_norm_rms_epsilon",
]

# Fields the writer regenerates itself (or that are reader-only virtual fields);
# copying them would raise a duplicate-key warning/error.
SKIP_FIELDS = {gguf.Keys.General.ARCHITECTURE}  # plus any "GGUF.*" prefix.


def field_value_and_types(field: gguf.ReaderField):
    """Return (value, value_type, sub_type) for a reader field, mirroring
    gguf_new_metadata.copy_with_new_metadata."""
    val_type = field.types[0]
    sub_type = field.types[-1] if val_type == GGUFValueType.ARRAY else None
    return field.contents(), val_type, sub_type


def copy_metadata(reader: GGUFReader, writer: GGUFWriter) -> int:
    """Copy every KV field from reader to writer verbatim. Returns the number of
    fields copied. Raises on anything unexpected so problems are loud."""
    copied = 0
    skipped = []
    for field in reader.fields.values():
        if field.name in SKIP_FIELDS or field.name.startswith("GGUF."):
            skipped.append(field.name)
            continue
        value, val_type, sub_type = field_value_and_types(field)
        if value is None:
            skipped.append(field.name)
            continue
        writer.add_key_value(field.name, value, val_type, sub_type=sub_type)
        copied += 1
    if skipped:
        print(f"  (skipped {len(skipped)} reader-managed/empty fields: {skipped})")
    return copied


def dequantize_tensor(tensor: gguf.ReaderTensor) -> np.ndarray:
    """Return a float32 numpy array in the SAME numpy (row-major) layout as the
    source tensor.data, so add_tensor reproduces identical ne[]."""
    if tensor.tensor_type == GGMLQuantizationType.F32:
        return np.asarray(tensor.data, dtype=np.float32)
    f32 = gguf.quants.dequantize(tensor.data, tensor.tensor_type)
    return np.asarray(f32, dtype=np.float32)


def self_verify(reader: GGUFReader) -> None:
    """Re-open the written file and assert it matches the source."""
    print("Self-verifying output ...")
    out = GGUFReader(OUT_PATH, "r")

    # (a) tensor count.
    assert len(out.tensors) == len(reader.tensors), (
        f"tensor count mismatch: out={len(out.tensors)} src={len(reader.tensors)}"
    )
    print(f"  tensor count matches: {len(out.tensors)}")

    # (b) tokenizer vocab length.
    src_tokens = reader.get_field(gguf.Keys.Tokenizer.LIST)
    out_tokens = out.get_field(gguf.Keys.Tokenizer.LIST)
    assert src_tokens is not None and out_tokens is not None, "tokens field missing"
    src_n = len(src_tokens.data)
    out_n = len(out_tokens.data)
    assert src_n == out_n, f"token count mismatch: out={out_n} src={src_n}"
    print(f"  tokenizer.ggml.tokens length matches: {out_n}")

    # (c) key llama.* scalars.
    for key in HYPERPARAM_KEYS:
        s = reader.get_field(key)
        o = out.get_field(key)
        assert s is not None and o is not None, f"missing scalar {key}"
        sv, ov = s.contents(), o.contents()
        if isinstance(sv, float):
            assert abs(sv - ov) < 1e-9, f"{key}: out={ov} src={sv}"
        else:
            assert sv == ov, f"{key}: out={ov} src={sv}"
    print(f"  {len(HYPERPARAM_KEYS)} llama.* scalars match")

    # (d) value/shape spot-check on 3 differently-shaped tensors.
    out_by_name = {t.name: t for t in out.tensors}
    # Pick tensors with distinct shapes: 1-D norm, 2-D square-ish, 2-D vocab.
    picks = []
    seen_shapes = set()
    preferred = [
        "token_embd.weight",
        "blk.0.attn_q.weight",
        "blk.0.ffn_gate.weight",
        "output_norm.weight",
        "blk.0.attn_norm.weight",
    ]
    name_to_src = {t.name: t for t in reader.tensors}
    for name in preferred:
        t = name_to_src.get(name)
        if t is None:
            continue
        shp = tuple(t.shape.tolist())
        if shp in seen_shapes:
            continue
        seen_shapes.add(shp)
        picks.append(t)
        if len(picks) == 3:
            break
    assert len(picks) == 3, "could not select 3 distinct-shape tensors"

    for t in picks:
        out_t = out_by_name[t.name]
        # Logical shapes (ggml ne[]) must be identical. NOTE: for quantized
        # source tensors, t.data.shape is the *block-encoded byte* shape (e.g.
        # Q8_0 packs 2048 f32 into 2176 bytes), so we compare against the
        # DEQUANTIZED reference shape, not t.data.shape.
        assert list(out_t.shape) == list(t.shape), (
            f"{t.name}: ne mismatch out={list(out_t.shape)} src={list(t.shape)}"
        )
        ref = dequantize_tensor(t)
        got = np.asarray(out_t.data, dtype=np.float32)
        assert got.shape == ref.shape, (
            f"{t.name}: dequantized numpy shape mismatch "
            f"out={got.shape} ref={ref.shape}"
        )
        max_diff = float(np.max(np.abs(ref - got))) if ref.size else 0.0
        assert max_diff < 1e-3, f"{t.name}: max value diff {max_diff} >= 1e-3"
        print(
            f"  spot-check {t.name}: ne={list(out_t.shape)} "
            f"src_type={t.tensor_type.name} max|diff|={max_diff:.2e} OK"
        )

    print("Self-verify PASSED.")


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {SRC_FILENAME} from {REPO_ID} ...")
    src_path = hf_hub_download(
        repo_id=REPO_ID, filename=SRC_FILENAME, local_dir=str(MODELS_DIR)
    )
    print(f"  source: {src_path} ({Path(src_path).stat().st_size / 1e9:.2f} GB)")

    print("Opening source with GGUFReader ...")
    reader = GGUFReader(src_path, "r")
    print(f"  {len(reader.fields)} KV fields, {len(reader.tensors)} tensors")

    print(f"Creating writer for {OUT_PATH} (arch={ARCH!r}) ...")
    writer = GGUFWriter(str(OUT_PATH), arch=ARCH)

    # Preserve any custom alignment from the source (matches gguf_new_metadata).
    align = reader.get_field(gguf.Keys.General.ALIGNMENT)
    if align is not None:
        writer.data_alignment = align.contents()
        print(f"  using source alignment: {writer.data_alignment}")

    print("Copying metadata ...")
    n_copied = copy_metadata(reader, writer)
    print(f"  copied {n_copied} KV fields")

    print("Dequantizing tensors to F32 ...")
    for i, tensor in enumerate(reader.tensors):
        f32 = dequantize_tensor(tensor)
        writer.add_tensor(tensor.name, f32)
        if i % 25 == 0 or i == len(reader.tensors) - 1:
            print(
                f"  [{i + 1}/{len(reader.tensors)}] {tensor.name} "
                f"({tensor.tensor_type.name} -> F32) ne={list(tensor.shape)}"
            )

    print("Writing output file ...")
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file(progress=True)
    writer.close()

    size = OUT_PATH.stat().st_size
    print(f"Wrote {OUT_PATH} ({size / 1e9:.3f} GB)")

    self_verify(reader)


if __name__ == "__main__":
    sys.exit(main())
