#!/usr/bin/env python3
"""Tiny tokenizer-only GGUF fixture for the C++ tokenizer test.

The real checkpoint (models/tinyllama-1.1b-chat-f32.gguf) is 4.4 GB and
gitignored, so tests/test_tokenizer.cpp can't load it. This tool copies ONLY the
tokenizer metadata -- tokenizer.ggml.{model,tokens,scores,token_type} plus the
bos/eos/unk token ids and general.architecture -- into a small standalone GGUF
(`tests/gguf/tokenizer.gguf`, a few MB, no tensors). That's everything
Tokenizer::from_gguf needs.

It reads the source with the canonical gguf.GGUFReader and writes the fixture
with gguf.GGUFWriter, so the fixture is a real, reader-validated GGUF.

The fixture is gitignored (tests/gguf/ is). Regenerate with:
  python3 tools/gen_tokenizer_fixture.py [path/to/source.gguf]
The C++ test SKIPs if the fixture is absent.
"""

import sys
from pathlib import Path

import gguf

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SRC = ROOT / "models" / "tinyllama-1.1b-chat-f32.gguf"
OUT_DIR = ROOT / "tests" / "gguf"
OUT_PATH = OUT_DIR / "tokenizer.gguf"


def main():
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_SRC
    if not src.exists():
        raise SystemExit(f"source GGUF not found: {src}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    reader = gguf.GGUFReader(str(src))

    def get(key):
        f = reader.fields.get(key)
        if f is None:
            raise SystemExit(f"source GGUF missing required key: {key}")
        return f.contents()

    arch = get("general.architecture")
    model = get("tokenizer.ggml.model")
    tokens = get("tokenizer.ggml.tokens")
    scores = [float(x) for x in get("tokenizer.ggml.scores")]
    token_type = [int(x) for x in get("tokenizer.ggml.token_type")]
    bos = int(get("tokenizer.ggml.bos_token_id"))
    eos = int(get("tokenizer.ggml.eos_token_id"))
    unk = int(get("tokenizer.ggml.unknown_token_id"))

    assert len(tokens) == len(scores) == len(token_type), "tokens/scores/token_type length mismatch"

    # Some GGUF conversions drop the SentencePiece scores (every entry == 0.0).
    # The SPM merge is score-based, so zeroed scores make encode() degenerate and
    # diverge from sentencepiece. Recover the real scores from tokenizer.model when
    # available -- the fixture is meant to match the sentencepiece reference.
    if all(x == 0.0 for x in scores):
        spm_model = ROOT / "models" / "tok" / "tokenizer.model"
        if spm_model.exists():
            import sentencepiece as spm

            sp = spm.SentencePieceProcessor()
            sp.load(str(spm_model))
            if sp.get_piece_size() == len(tokens):
                scores = [float(sp.get_score(i)) for i in range(len(tokens))]
                print(f"  NOTE: GGUF scores were all-zero; restored from {spm_model}")
            else:
                print("  WARNING: tokenizer.model vocab size differs; keeping zero scores")
        else:
            print("  WARNING: GGUF scores are all-zero and tokenizer.model is absent; "
                  "encode() will not match sentencepiece")

    writer = gguf.GGUFWriter(str(OUT_PATH), arch=str(arch))
    writer.add_tokenizer_model(str(model))
    writer.add_token_list(list(tokens))
    writer.add_token_scores(scores)
    writer.add_token_types(token_type)
    writer.add_bos_token_id(bos)
    writer.add_eos_token_id(eos)
    writer.add_unk_token_id(unk)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()  # no tensors, but finalizes the file
    writer.close()

    size_mb = OUT_PATH.stat().st_size / (1024 * 1024)
    print(f"wrote {OUT_PATH}  ({size_mb:.1f} MB)")
    print(f"  arch={arch} model={model} vocab={len(tokens)}")
    print(f"  bos={bos} eos={eos} unk={unk}")


if __name__ == "__main__":
    main()
