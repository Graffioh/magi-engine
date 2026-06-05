#!/usr/bin/env python3
"""Tiny GGUF fixture generator for the C++ GGUF loader tests. - by opus

Writes a minimal F32 GGUF (`tests/gguf/tiny.gguf`) plus the *source* arrays as
raw little-endian float32 blobs (`tests/gguf/<name>.f32`) and a `manifest.txt`,
so the C++ round-trip test (tests/test_gguf.cpp) can compare the loaded mmap
views byte-for-byte against the exact arrays we wrote.

Why two artifacts: the .gguf exercises the *reader* (header, typed metadata KV
block, tensor directory, aligned data section, reversed ne[] dims). The .f32
blobs are the trusted source values the views must reproduce -- the C++ test
loads them the same way the golden harness does and compares with check().

Writer path: we strongly prefer the *canonical* `gguf.GGUFWriter`, because that
validates our hand-written reader against the real llama.cpp format rather than
against our own assumptions. If `gguf` isn't importable we try a one-shot
`pip install gguf`; if that fails (offline) we fall back to a small pure-Python
writer that emits the exact byte layout the reader expects. The chosen path is
printed.

Metadata covers multiple KV types on purpose (u32, f32, bool, string, and a
STRING ARRAY) to exercise every branch of the C++ KV parser. Tensors are varied
shapes (one 2D, one 1D) under llama-style names. Fixed seed 0 -> reproducible.

Run:  python3 tools/gen_tiny_gguf.py
"""

import struct
import subprocess
import sys
from pathlib import Path

import numpy as np

# ----------------------------------------------------------------------------
# Fixture definition. Fixed seed -> fully reproducible. Tensors are small and
# varied (a 2D weight and a 1D norm) so the reversed-ne[] shape logic is tested.
# ----------------------------------------------------------------------------
SEED = 0

OUT_DIR = Path(__file__).resolve().parent.parent / "tests" / "gguf"
GGUF_PATH = OUT_DIR / "tiny.gguf"

ARCH = "llama"  # general.architecture
ALIGNMENT = 32  # general.alignment (canonical gguf default; we set it explicitly)

# Metadata the C++ test asserts on. Keep keys/values in sync with test_gguf.cpp.
META_U32 = ("magi.test.u32", 1234)
META_F32 = ("magi.test.f32", 2.5)
META_BOOL = ("magi.test.bool", True)
META_STR = ("general.architecture", ARCH)
META_TOKENS = ("magi.test.tokens", ["<unk>", "<s>", "</s>", "hi"])


def make_tensors():
    """Source arrays (name -> float32 ndarray). At least one 2D and one 1D."""
    rng = np.random.default_rng(SEED)

    def randn(*shape):
        return (rng.standard_normal(shape) * 0.5).astype(np.float32)

    return {
        "blk.0.attn_q.weight": randn(4, 3),  # 2D: numpy (out=4, in=3)
        "output_norm.weight": randn(5),      # 1D: (5,)
    }


# ----------------------------------------------------------------------------
# Canonical writer: use gguf.GGUFWriter so we test the reader against the real
# format. add_tensor takes row-major numpy arrays and stores ne[] reversed, which
# is exactly what our reader un-reverses.
# ----------------------------------------------------------------------------
def write_with_canonical(tensors):
    import gguf

    # Passing arch="llama" makes the writer emit general.architecture="llama" for
    # us, so we don't add META_STR again (that would just overwrite with the same
    # value and warn). The C++ test still asserts general.architecture == "llama".
    writer = gguf.GGUFWriter(str(GGUF_PATH), arch=ARCH)
    writer.add_custom_alignment(ALIGNMENT)

    writer.add_uint32(*META_U32)
    writer.add_float32(*META_F32)
    writer.add_bool(*META_BOOL)
    writer.add_array(META_TOKENS[0], META_TOKENS[1])

    for name, arr in tensors.items():
        writer.add_tensor(name, arr)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


# ----------------------------------------------------------------------------
# Hand-rolled writer (offline fallback): emits the exact little-endian byte
# layout the C++ reader parses. Kept tiny and literal so it doubles as a spec
# reference. ne[] is stored innermost-first, i.e. numpy shape reversed.
# ----------------------------------------------------------------------------
# GGUF value-type tags.
VT_UINT32 = 4
VT_FLOAT32 = 6
VT_BOOL = 7
VT_STRING = 8
VT_ARRAY = 9
GGML_TYPE_F32 = 0
GGUF_MAGIC = 0x46554747
GGUF_VERSION = 3


def _u32(v):
    return struct.pack("<I", v)


def _u64(v):
    return struct.pack("<Q", v)


def _f32(v):
    return struct.pack("<f", v)


def _gstr(s):
    b = s.encode("utf-8")
    return _u64(len(b)) + b


def _kv_u32(key, val):
    return _gstr(key) + _u32(VT_UINT32) + _u32(val)


def _kv_f32(key, val):
    return _gstr(key) + _u32(VT_FLOAT32) + _f32(val)


def _kv_bool(key, val):
    return _gstr(key) + _u32(VT_BOOL) + struct.pack("<B", 1 if val else 0)


def _kv_string(key, val):
    return _gstr(key) + _u32(VT_STRING) + _gstr(val)


def _kv_str_array(key, vals):
    body = _u32(VT_ARRAY) + _u32(VT_STRING) + _u64(len(vals))
    for v in vals:
        body += _gstr(v)
    return _gstr(key) + body


def write_with_handrolled(tensors):
    # Build metadata KV block (order is irrelevant to the reader's hash map).
    kv = b"".join(
        [
            _kv_u32(*META_U32),
            _kv_f32(*META_F32),
            _kv_bool(*META_BOOL),
            _kv_string(*META_STR),
            _kv_str_array(*META_TOKENS),
            _kv_u32("general.alignment", ALIGNMENT),
        ]
    )
    kv_count = 6

    # Tensor directory. Offsets are relative to the (aligned) data section start
    # and must themselves be multiples of ALIGNMENT. Lay tensors out back to back,
    # padding each to the alignment.
    names = list(tensors.keys())
    directory = b""
    data = b""
    for name in names:
        arr = np.ascontiguousarray(tensors[name], dtype="<f4")
        ne = list(arr.shape)[::-1]  # innermost-first
        directory += _gstr(name)
        directory += _u32(len(ne))
        for d in ne:
            directory += _u64(d)
        directory += _u32(GGML_TYPE_F32)
        directory += _u64(len(data))  # offset within data section
        data += arr.tobytes()
        # pad up to alignment so the next tensor's offset is aligned
        pad = (-len(data)) % ALIGNMENT
        data += b"\x00" * pad

    header = _u32(GGUF_MAGIC) + _u32(GGUF_VERSION) + _u64(len(names)) + _u64(kv_count)

    pre_data = header + kv + directory
    pad = (-len(pre_data)) % ALIGNMENT  # align the data section start
    blob = pre_data + b"\x00" * pad + data

    GGUF_PATH.write_bytes(blob)


# ----------------------------------------------------------------------------
# Source-blob serialization + round-trip self-check.
# ----------------------------------------------------------------------------
def write_source_blobs(tensors):
    manifest = ["# tiny.gguf source arrays", "# name  shape..."]
    for name, arr in tensors.items():
        arr.astype("<f4").tofile(OUT_DIR / f"{name}.f32")
        manifest.append(f"{name} " + " ".join(str(d) for d in arr.shape))
    (OUT_DIR / "manifest.txt").write_text("\n".join(manifest) + "\n")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tensors = make_tensors()

    # Decide the writer path: canonical gguf strongly preferred.
    path_used = None
    try:
        import gguf  # noqa: F401

        path_used = "canonical gguf"
    except ImportError:
        print("gguf not importable; attempting a one-shot `pip install gguf`...")
        try:
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "--quiet", "gguf"],
                check=True,
            )
            import gguf  # noqa: F401

            path_used = "canonical gguf (pip-installed)"
        except Exception as e:
            print(f"  pip install failed ({e}); falling back to hand-rolled writer.")
            path_used = "hand-rolled (offline fallback)"

    if path_used.startswith("canonical"):
        write_with_canonical(tensors)
    else:
        write_with_handrolled(tensors)

    write_source_blobs(tensors)

    # Round-trip self-check: the .f32 blobs must decode back to the exact arrays.
    for name, arr in tensors.items():
        back = np.fromfile(OUT_DIR / f"{name}.f32", dtype="<f4").reshape(arr.shape)
        assert np.array_equal(back, arr.astype(np.float32)), f"round-trip mismatch: {name}"

    # If we hand-rolled the GGUF, sanity-check the header bytes we just wrote.
    if path_used.startswith("hand-rolled"):
        raw = GGUF_PATH.read_bytes()
        magic, version, n_tensors, n_kv = struct.unpack("<IIQQ", raw[:24])
        assert magic == GGUF_MAGIC, "hand-rolled magic mismatch"
        assert version == GGUF_VERSION, "hand-rolled version mismatch"
        assert n_tensors == len(tensors), "hand-rolled tensor count mismatch"

    print(f"writer path: {path_used}")
    print(f"wrote {GGUF_PATH}")
    print(f"  {len(tensors)} tensors, source blobs + manifest in {OUT_DIR}")
    for name, arr in tensors.items():
        print(f"  {name:24s} shape={arr.shape}  first={arr.flatten()[0]:.6f}")
    print(f"  metadata: {META_U32[0]}={META_U32[1]}, {META_F32[0]}={META_F32[1]}, "
          f"{META_BOOL[0]}={META_BOOL[1]}, {META_STR[0]}={META_STR[1]!r}, "
          f"{META_TOKENS[0]}={META_TOKENS[1]}")


if __name__ == "__main__":
    main()
