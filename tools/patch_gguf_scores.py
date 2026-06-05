#!/usr/bin/env python3
"""Patch zeroed tokenizer.ggml.scores in a GGUF in place from tokenizer.model.

The TheBloke Q8_0 GGUF this project converts from ships with an all-zero
tokenizer.ggml.scores array (the SentencePiece scores were dropped in
conversion), and make_f32_gguf.py copies that verbatim. The SPM merge is
score-based, so encode() can't match sentencepiece with zero scores.

This rewrites ONLY the FLOAT32[vocab] scores block in place -- same element
count, same byte offsets -- with the genuine scores read from the original
SentencePiece model (models/tok/tokenizer.model). No tensor data is touched and
the file size is unchanged, so it's safe to run on the multi-GB checkpoint.

Run:  python3 tools/patch_gguf_scores.py [path/to/model.gguf]
"""

import struct
import sys
from pathlib import Path

import gguf
import sentencepiece as spm

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_GGUF = ROOT / "models" / "tinyllama-1.1b-chat-f32.gguf"
SPM_MODEL = ROOT / "models" / "tok" / "tokenizer.model"


def main():
    gguf_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_GGUF
    if not gguf_path.exists():
        raise SystemExit(f"GGUF not found: {gguf_path}")
    if not SPM_MODEL.exists():
        raise SystemExit(f"tokenizer.model not found: {SPM_MODEL}")

    reader = gguf.GGUFReader(str(gguf_path))
    f = reader.fields.get("tokenizer.ggml.scores")
    if f is None:
        raise SystemExit("GGUF has no tokenizer.ggml.scores field")

    scores = f.contents()
    n = len(scores)
    if any(x != 0.0 for x in scores):
        print(f"scores already non-zero ({sum(1 for x in scores if x!=0.0)}/{n}); nothing to do")
        return

    # Byte offset of the contiguous FLOAT32 value block within the file.
    base_addr = reader.data.ctypes.data
    first_part = f.parts[f.data[0]]
    last_part = f.parts[f.data[-1]]
    off0 = first_part.ctypes.data - base_addr
    off1 = last_part.ctypes.data - base_addr
    assert first_part.dtype.itemsize == 4, "expected float32 scores"
    assert off1 - off0 == (n - 1) * 4, "scores block is not contiguous"

    sp = spm.SentencePieceProcessor()
    sp.load(str(SPM_MODEL))
    if sp.get_piece_size() != n:
        raise SystemExit(f"vocab mismatch: gguf={n} tokenizer.model={sp.get_piece_size()}")

    new_scores = [float(sp.get_score(i)) for i in range(n)]
    blob = struct.pack(f"<{n}f", *new_scores)
    assert len(blob) == n * 4

    # Drop the read-only mmap before reopening for writing.
    del reader

    with open(gguf_path, "r+b") as fh:
        fh.seek(off0)
        fh.write(blob)

    print(f"patched {n} scores in {gguf_path} at byte offset {off0}")
    print(f"  sample: score[274]={new_scores[274]:.1f} score[450]={new_scores[450]:.1f}")


if __name__ == "__main__":
    main()
