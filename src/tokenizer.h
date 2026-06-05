#pragma once

#include "gguf.h"

#include <array>
#include <string>
#include <unordered_map>
#include <vector>

// SentencePiece (llama SPM) tokenizer, built entirely from a GGUF's
// tokenizer.ggml.* metadata. This is the byte-pair-free, *score-based* unigram
// merge variant llama.cpp uses for Llama/TinyLlama -- NOT a merges-rank BPE.
//
// encode() reproduces llama.cpp's algorithm exactly so our ids match the real
// `sentencepiece` reference token-for-token:
//   1. prepend a dummy space, then map every ASCII space -> the meta-space
//      U+2581 ("▁").
//   2. split into one-codepoint symbols held in a doubly-linked list.
//   3. greedily merge the highest-*score* adjacent bigram (max-heap), llama.cpp
//      style, re-seeding neighbours after each merge.
//   4. emit token ids; symbols with no vocab entry fall back to their raw bytes
//      via the "<0xXX>" byte tokens.
//
// decode() inverts this: concatenate pieces, turn the meta-space back into
// spaces, expand byte tokens into raw bytes, and trim the one dummy-prefix space.
class Tokenizer {
  public:
    // Build from the GGUF metadata (tokens / scores / token_type + special ids).
    // Throws std::runtime_error if the required tokenizer.ggml.* keys are absent
    // or the model isn't an SPM ("llama") tokenizer.
    static Tokenizer from_gguf(const gguf::GGUF& g);

    // Token ids for `text`. With add_bos, bos_id() is prepended.
    std::vector<int> encode(const std::string& text, bool add_bos) const;

    // Decode a full id sequence to text (skips BOS, stops at EOS, trims the one
    // leading dummy space). Byte tokens are coalesced into raw UTF-8 bytes.
    std::string decode(const std::vector<int>& ids) const;

    // Decode a single piece (for streaming). No dummy-space trimming. A lone
    // byte token yields that single raw byte (may be an incomplete UTF-8 unit).
    std::string decode_token(int id) const;

    int bos_id() const { return bos_id_; }
    int eos_id() const { return eos_id_; }
    int unk_id() const { return unk_id_; }

    int vocab_size() const { return static_cast<int>(tokens_.size()); }

  private:
    std::vector<std::string>             tokens_;       // id -> piece text
    std::vector<float>                   scores_;       // id -> merge score
    std::unordered_map<std::string, int> token_to_id_;  // piece text -> id

    // Byte-fallback: raw byte value (0..255) -> id of its "<0xXX>" token, or -1
    // if that byte token is absent from the vocab.
    std::array<int, 256> byte_to_id_{};

    int bos_id_ = 1;
    int eos_id_ = 2;
    int unk_id_ = 0;
};
