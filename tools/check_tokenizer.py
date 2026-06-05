#!/usr/bin/env python3
"""Reference SPM ids from the real `sentencepiece`, for the C++ tokenizer test.

Loads the Llama tokenizer (`models/tok/tokenizer.model`, fetched from
TinyLlama/TinyLlama-1.1B-Chat-v1.0) with the genuine SentencePiece library and
prints the token ids it produces for a batch of fixtures -- the ground truth our
hand-rolled `Tokenizer::encode` must reproduce exactly (no BOS).

It also writes those ids to `tools/tokenizer_reference.json` so other tooling can
diff against them, and prints C++-literal-ready lines so the ids can be baked
into tests/test_tokenizer.cpp.

Setup:
  pip install sentencepiece huggingface_hub
  python3 -c "from huggingface_hub import hf_hub_download as d; \
             d('TinyLlama/TinyLlama-1.1B-Chat-v1.0','tokenizer.model',local_dir='models/tok')"
Run:
  python3 tools/check_tokenizer.py
"""

import json
from pathlib import Path

import sentencepiece as spm

ROOT = Path(__file__).resolve().parent.parent
MODEL = ROOT / "models" / "tok" / "tokenizer.model"
OUT_JSON = ROOT / "tools" / "tokenizer_reference.json"

# The fixtures. Mix of plain text, punctuation, code (newlines + indentation),
# a model-name string, and unicode (accents, an emoji, CJK) to exercise the
# meta-space handling and the byte-fallback path.
TEST_STRINGS = [
    "The capital of France is",
    "Hello, world!",
    "def foo(x):\n    return x*2",
    "TinyLlama is a 1.1B model.",
    "café ☕ 日本語",
]


def main():
    if not MODEL.exists():
        raise SystemExit(
            f"missing {MODEL}\n"
            "download it first:\n"
            '  python3 -c "from huggingface_hub import hf_hub_download as d; '
            "d('TinyLlama/TinyLlama-1.1B-Chat-v1.0','tokenizer.model',local_dir='models/tok')\""
        )

    sp = spm.SentencePieceProcessor()
    sp.load(str(MODEL))
    print(f"loaded {MODEL}  (vocab_size={sp.get_piece_size()})")
    print(f"bos={sp.bos_id()} eos={sp.eos_id()} unk={sp.unk_id()}\n")

    out = {}
    for s in TEST_STRINGS:
        ids = sp.encode(s, out_type=int)  # no BOS/EOS
        pieces = [sp.id_to_piece(i) for i in ids]
        out[s] = ids
        print(f"string : {s!r}")
        print(f"  ids   : {ids}")
        print(f"  pieces: {pieces}")
        # C++-literal-ready line for tests/test_tokenizer.cpp.
        print("  c++   : { " + ", ".join(str(i) for i in ids) + " }")
        print()

    OUT_JSON.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"wrote reference ids -> {OUT_JSON}")


if __name__ == "__main__":
    main()
