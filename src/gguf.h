#pragma once

#include "mmap.h"
#include "tensor.h"

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

// GGUF loader. GGUF is the on-disk container llama.cpp uses for model weights +
// metadata: a small header, a typed key/value metadata block, a tensor directory,
// then the raw tensor data (mmap-friendly, aligned). We only support F32 tensors
// here -- enough to load our own toy models and any pre-quant F32 checkpoint.
//
// The file stays mmapped for the lifetime of the GGUF: every Tensor below is a
// *view* into the mapping (no copy), kept alive by sharing the same
// shared_ptr<MappedFile> as its owner_. So GGUF owns move-only Tensors and is
// itself move-only -- return it by value and let NRVO/move do the work.
namespace gguf {

// Minimal typed metadata value. We don't need to distinguish u8/u16/.../i64 in
// the engine, so every integer kv is widened to int64 and f32/f64 to double;
// the typed accessors below (i64/f64/u32/f32) narrow back at the call site.
// Arrays are never nested in GGUF, so the variant only holds flat vectors.
using MetadataValue = std::variant<std::int64_t,
                                   double,
                                   bool,
                                   std::string,
                                   std::vector<std::int64_t>,
                                   std::vector<double>,
                                   std::vector<std::string>>;

struct GGUF {
    std::shared_ptr<MappedFile>                    mapping;       // keeps the mmap alive
    std::uint32_t                                  version = 0;   // 2 or 3
    std::unordered_map<std::string, MetadataValue> metadata;      // typed kv block
    std::unordered_map<std::string, Tensor>        tensors;       // mmap views (reversed shapes)

    // GGUF holds move-only Tensors, so it is move-only too.
    GGUF()                       = default;
    GGUF(const GGUF&)            = delete;
    GGUF& operator=(const GGUF&) = delete;
    GGUF(GGUF&&)                 = default;
    GGUF& operator=(GGUF&&)      = default;

    bool has(const std::string& key) const { return metadata.count(key) != 0; }

    // Typed accessors. i64/f64 accept any integer/float kv (we widened on load);
    // u32/f32 just narrow those. They throw if the key is missing or the stored
    // alternative doesn't match, so a bad config surfaces loudly instead of as UB.
    std::int64_t i64(const std::string& key) const;  // works for any int kv
    double       f64(const std::string& key) const;  // works for f32/f64 kv

    std::uint32_t u32(const std::string& key) const { return static_cast<std::uint32_t>(i64(key)); }

    float f32(const std::string& key) const { return static_cast<float>(f64(key)); }

    bool                            boolean(const std::string& key) const;
    const std::string&              str(const std::string& key) const;
    const std::vector<std::string>& str_array(const std::string& key) const;

    // Array accessors for the numeric tokenizer metadata: scores arrive as a
    // float array (widened to double on load) and token_type as an int array
    // (widened to int64). Mirror str_array -- throw on missing/wrong type.
    const std::vector<double>&       f64_array(const std::string& key) const;
    const std::vector<std::int64_t>& i64_array(const std::string& key) const;

    const Tensor& tensor(const std::string& name) const;
};

// Memory-map `path` and parse it into a GGUF. Throws std::runtime_error on a bad
// magic/version, a truncated stream, an unsupported (non-F32) tensor, or any
// tensor whose data would run past the end of the file.
GGUF load_gguf(const std::string& path);

}  // namespace gguf
