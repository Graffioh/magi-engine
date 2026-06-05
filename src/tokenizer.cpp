#include "tokenizer.h"

#include <cstdint>
#include <cstdio>
#include <queue>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

// The SentencePiece meta-space U+2581 ("LOWER ONE EIGHTH BLOCK"), UTF-8 encoded.
// llama.cpp replaces every ASCII space with this 3-byte sequence before merging.
const std::string kMetaSpace = "\xe2\x96\x81";  // 0xE2 0x96 0x81

// Length in bytes of the UTF-8 codepoint starting at lead byte `c`. SentencePiece
// splits the (already meta-space-substituted) text into one-codepoint symbols, so
// we need to walk codepoint-by-codepoint. A malformed/continuation lead byte is
// treated as a single byte so we never read past the symbol.
int utf8_len(unsigned char c) {
    if (c < 0x80) {
        return 1;
    }
    if ((c >> 5) == 0x06) {  // 110xxxxx
        return 2;
    }
    if ((c >> 4) == 0x0e) {  // 1110xxxx
        return 3;
    }
    if ((c >> 3) == 0x1e) {  // 11110xxx
        return 4;
    }
    return 1;  // continuation or invalid lead -> consume one byte
}

}  // namespace

Tokenizer Tokenizer::from_gguf(const gguf::GGUF& g) {
    Tokenizer t;

    // We only implement the llama SPM (SentencePiece) path. BPE models would need
    // a different merge algorithm, so fail loudly rather than silently mis-encode.
    if (g.has("tokenizer.ggml.model")) {
        const std::string& model = g.str("tokenizer.ggml.model");
        if (model != "llama") {
            throw std::runtime_error("tokenizer: unsupported tokenizer.ggml.model '" + model +
                                     "' (only llama/SPM is supported)");
        }
    }

    const std::vector<std::string>& tokens = g.str_array("tokenizer.ggml.tokens");
    const std::vector<double>&      scores = g.f64_array("tokenizer.ggml.scores");
    if (tokens.size() != scores.size()) {
        throw std::runtime_error("tokenizer: tokens/scores length mismatch");
    }

    t.tokens_ = tokens;
    t.scores_.reserve(scores.size());
    t.token_to_id_.reserve(scores.size());
    for (std::size_t id = 0; id < tokens.size(); ++id) {
        t.scores_.push_back(static_cast<float>(scores[id]));
        // First-wins on duplicate pieces (matches llama.cpp, where lower ids are
        // the canonical entries). insert() is a no-op if the key already exists.
        t.token_to_id_.emplace(tokens[id], static_cast<int>(id));
    }

    // Byte-fallback table: SPM byte tokens are "<0xXX>" (uppercase hex), one per
    // raw byte value. Look each up by its literal piece text. Absent -> -1.
    t.byte_to_id_.fill(-1);
    for (int b = 0; b < 256; ++b) {
        char buf[8];
        std::snprintf(buf, sizeof(buf), "<0x%02X>", b);
        auto it = t.token_to_id_.find(buf);
        if (it != t.token_to_id_.end()) {
            t.byte_to_id_[static_cast<std::size_t>(b)] = it->second;
        }
    }

    // Special ids. Default to the Llama convention if a key is missing.
    if (g.has("tokenizer.ggml.bos_token_id")) {
        t.bos_id_ = static_cast<int>(g.i64("tokenizer.ggml.bos_token_id"));
    }
    if (g.has("tokenizer.ggml.eos_token_id")) {
        t.eos_id_ = static_cast<int>(g.i64("tokenizer.ggml.eos_token_id"));
    }
    if (g.has("tokenizer.ggml.unknown_token_id")) {
        t.unk_id_ = static_cast<int>(g.i64("tokenizer.ggml.unknown_token_id"));
    }

    return t;
}

std::vector<int> Tokenizer::encode(const std::string& text, bool add_bos) const {
    // --- Step 1: preprocess. Prepend one dummy space, then map every ASCII space
    // to the meta-space. "Hello world" -> "▁Hello▁world". This dummy prefix is why
    // most leading tokens start with "▁" and why decode trims one leading space. ---
    std::string normalized;
    normalized.reserve(text.size() + kMetaSpace.size());
    normalized += kMetaSpace;  // the dummy prefix space, as a meta-space
    for (char c : text) {
        if (c == ' ') {
            normalized += kMetaSpace;
        } else {
            normalized += c;
        }
    }

    // --- Step 2: split into one-codepoint symbols in a doubly-linked list. Each
    // symbol is a view (offset+len) into `normalized` plus prev/next indices. We
    // track `n` (current byte length) per symbol so a merge just grows the left
    // symbol and unlinks the right; a stale heap entry is detected via length. ---
    struct Symbol {
        int         prev;
        int         next;
        std::size_t offset;
        std::size_t n;  // byte length; 0 once merged away
    };
    std::vector<Symbol> symbols;
    symbols.reserve(normalized.size());
    for (std::size_t i = 0; i < normalized.size();) {
        const int   len = utf8_len(static_cast<unsigned char>(normalized[i]));
        std::size_t cp  = static_cast<std::size_t>(len);
        if (i + cp > normalized.size()) {
            cp = normalized.size() - i;  // clamp a truncated trailing codepoint
        }
        const int idx = static_cast<int>(symbols.size());
        symbols.push_back(Symbol{ idx - 1, idx + 1, i, cp });
        i += cp;
    }
    if (symbols.empty()) {
        std::vector<int> out;
        if (add_bos) {
            out.push_back(bos_id_);
        }
        return out;
    }
    symbols.back().next = -1;  // last symbol has no successor

    // --- Step 3: greedy score-based merge. A bigram records the two endpoints,
    // the merged piece's score (= priority), and the merged byte length at the
    // time it was queued. The max-heap pops highest score first; ties go to the
    // earlier (smaller-left-offset) bigram, matching SentencePiece/llama.cpp. ---
    struct Bigram {
        int         left;
        int         right;
        float       score;
        std::size_t len;     // merged byte length when queued (staleness check)
        std::size_t offset;  // left symbol's offset, for deterministic tie-break
    };
    struct BigramCmp {
        bool operator()(const Bigram& a, const Bigram& b) const {
            // priority_queue is a max-heap on `<`: "a worse than b" so b surfaces.
            // Higher score wins; on a tie the smaller offset (earlier text) wins.
            if (a.score != b.score) {
                return a.score < b.score;
            }
            return a.offset > b.offset;
        }
    };
    std::priority_queue<Bigram, std::vector<Bigram>, BigramCmp> work;

    auto try_add_bigram = [&](int left, int right) {
        if (left == -1 || right == -1) {
            return;
        }
        const std::string merged =
            normalized.substr(symbols[left].offset, symbols[left].n + symbols[right].n);
        auto it = token_to_id_.find(merged);
        if (it == token_to_id_.end()) {
            return;
        }
        work.push(Bigram{ left, right, scores_[static_cast<std::size_t>(it->second)],
                          merged.size(), symbols[left].offset });
    };

    // Seed with every adjacent pair.
    for (int i = 1; i < static_cast<int>(symbols.size()); ++i) {
        try_add_bigram(i - 1, i);
    }

    while (!work.empty()) {
        Bigram b = work.top();
        work.pop();

        Symbol& left  = symbols[b.left];
        Symbol& right = symbols[b.right];

        // Skip if either endpoint was already merged away, or the pair's combined
        // length no longer matches what we queued (llama.cpp's staleness test).
        if (left.n == 0 || right.n == 0 || left.n + right.n != b.len) {
            continue;
        }

        // Merge right into left: grow left, kill right, relink left -> right.next.
        left.n += right.n;
        right.n = 0;
        left.next = right.next;
        if (right.next != -1) {
            symbols[right.next].prev = b.left;
        }

        // Re-seed the two new neighbouring pairs around the grown symbol.
        try_add_bigram(left.prev, b.left);
        try_add_bigram(b.left, left.next);
    }

    // --- Step 4: emit. Walk the surviving symbols left-to-right. A symbol whose
    // text is in the vocab emits that id; otherwise byte-fallback emits one
    // "<0xXX>" token per raw byte (or unk if a byte token is somehow missing). ---
    std::vector<int> out;
    if (add_bos) {
        out.push_back(bos_id_);
    }
    for (int i = 0; i != -1; i = symbols[i].next) {
        const Symbol&     sym   = symbols[i];
        const std::string piece = normalized.substr(sym.offset, sym.n);
        auto              it    = token_to_id_.find(piece);
        if (it != token_to_id_.end()) {
            out.push_back(it->second);
        } else {
            for (std::size_t k = 0; k < sym.n; ++k) {
                const unsigned char byte = static_cast<unsigned char>(normalized[sym.offset + k]);
                const int           bid  = byte_to_id_[byte];
                out.push_back(bid != -1 ? bid : unk_id_);
            }
        }
    }
    return out;
}

std::string Tokenizer::decode(const std::vector<int>& ids) const {
    std::string out;
    for (int id : ids) {
        if (id == bos_id_) {
            continue;  // BOS is never rendered
        }
        if (id == eos_id_) {
            break;  // stop the moment EOS appears
        }
        if (id < 0 || id >= static_cast<int>(tokens_.size())) {
            continue;
        }

        const std::string& piece = tokens_[static_cast<std::size_t>(id)];

        // Byte token "<0xXX>": emit the single raw byte it stands for. Detect it
        // by exact piece shape so a normal token that merely looks similar isn't
        // mistaken for one.
        if (piece.size() == 6 && piece[0] == '<' && piece[1] == '0' && piece[2] == 'x' &&
            piece[5] == '>') {
            const int hi = std::stoi(piece.substr(3, 2), nullptr, 16);
            out.push_back(static_cast<char>(hi));
            continue;
        }

        // Normal piece: meta-space U+2581 -> ASCII space, everything else verbatim.
        for (std::size_t i = 0; i < piece.size();) {
            if (i + kMetaSpace.size() <= piece.size() &&
                piece.compare(i, kMetaSpace.size(), kMetaSpace) == 0) {
                out.push_back(' ');
                i += kMetaSpace.size();
            } else {
                out.push_back(piece[i]);
                ++i;
            }
        }
    }

    // Trim exactly one leading space -- the dummy prefix space encode() injected.
    if (!out.empty() && out.front() == ' ') {
        out.erase(out.begin());
    }
    return out;
}

std::string Tokenizer::decode_token(int id) const {
    if (id < 0 || id >= static_cast<int>(tokens_.size())) {
        return std::string();
    }
    const std::string& piece = tokens_[static_cast<std::size_t>(id)];

    if (piece.size() == 6 && piece[0] == '<' && piece[1] == '0' && piece[2] == 'x' &&
        piece[5] == '>') {
        const int hi = std::stoi(piece.substr(3, 2), nullptr, 16);
        return std::string(1, static_cast<char>(hi));
    }

    std::string out;
    for (std::size_t i = 0; i < piece.size();) {
        if (i + kMetaSpace.size() <= piece.size() &&
            piece.compare(i, kMetaSpace.size(), kMetaSpace) == 0) {
            out.push_back(' ');
            i += kMetaSpace.size();
        } else {
            out.push_back(piece[i]);
            ++i;
        }
    }
    return out;
}
