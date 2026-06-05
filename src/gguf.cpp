#include "gguf.h"

#include "mmap.h"
#include "tensor.h"

#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

namespace gguf {

namespace {

// The four header constants. magic spells "GGUF" little-endian; we accept format
// versions 2 and 3 (they share this layout) and reject anything else loudly.
constexpr std::uint32_t GGUF_MAGIC        = 0x46554747;
constexpr std::uint64_t DEFAULT_ALIGNMENT = 32;
constexpr std::uint32_t GGML_TYPE_F32     = 0;  // the only tensor type we support

// GGUF metadata value-type tags (see the format spec). ARRAY nests one of the
// scalar tags below it; arrays are never themselves nested (elem_type != ARRAY).
enum ValueType : std::uint32_t {
    VT_UINT8   = 0,
    VT_INT8    = 1,
    VT_UINT16  = 2,
    VT_INT16   = 3,
    VT_UINT32  = 4,
    VT_INT32   = 5,
    VT_FLOAT32 = 6,
    VT_BOOL    = 7,
    VT_STRING  = 8,
    VT_ARRAY   = 9,
    VT_UINT64  = 10,
    VT_INT64   = 11,
    VT_FLOAT64 = 12,
};

// align_up(x, a) rounds x up to the next multiple of a. Used for the data-section
// start: GGUF pads the gap after the tensor directory so tensor data lands on an
// `alignment`-byte boundary (cheap aligned mmap reads).
std::uint64_t align_up(std::uint64_t x, std::uint64_t a) {
    return (x + a - 1) / a * a;
}

// A bounds-checked cursor over the mmapped bytes. Every read advances `cur_` and
// asserts (via throw) that it stays within [begin, end); the metadata stream is
// NOT aligned, so we memcpy into typed locals rather than deref unaligned bytes.
class Cursor {
  private:
    const std::byte* cur_;
    const std::byte* end_;

    void require(std::size_t n) const {
        if (n > static_cast<std::size_t>(end_ - cur_)) {
            throw std::runtime_error("gguf: unexpected end of file while parsing");
        }
    }

    // Copy `n` raw bytes into `dst` (an integer/float local) and advance.
    template <typename T>
    T read_raw() {
        require(sizeof(T));
        T v;
        std::memcpy(&v, cur_, sizeof(T));
        cur_ += sizeof(T);
        return v;
    }

  public:
    Cursor(const std::byte* begin, const std::byte* end) : cur_(begin), end_(end) {}

    const std::byte* pos() const { return cur_; }

    std::uint8_t  read_u8() { return read_raw<std::uint8_t>(); }
    std::int8_t   read_i8() { return read_raw<std::int8_t>(); }
    std::uint16_t read_u16() { return read_raw<std::uint16_t>(); }
    std::int16_t  read_i16() { return read_raw<std::int16_t>(); }
    std::uint32_t read_u32() { return read_raw<std::uint32_t>(); }
    std::int32_t  read_i32() { return read_raw<std::int32_t>(); }
    std::uint64_t read_u64() { return read_raw<std::uint64_t>(); }
    std::int64_t  read_i64() { return read_raw<std::int64_t>(); }
    float         read_f32() { return read_raw<float>(); }
    double        read_f64() { return read_raw<double>(); }

    // A GGUF string is a u64 length followed by exactly `len` raw bytes -- no
    // null terminator, so we build the std::string from the [ptr, ptr+len) range.
    std::string read_string() {
        std::uint64_t len = read_u64();
        require(len);
        std::string s(reinterpret_cast<const char*>(cur_), len);
        cur_ += len;
        return s;
    }
};

// Read one scalar metadata value of `type`, widening to the variant's storage:
// every integer -> int64 (signed types sign-extend), every float -> double.
// BOOL is one byte (0/1). Throws on ARRAY -- callers handle that case directly.
MetadataValue read_scalar(Cursor& c, std::uint32_t type) {
    switch (type) {
        case VT_UINT8:
            return static_cast<std::int64_t>(c.read_u8());
        case VT_INT8:
            return static_cast<std::int64_t>(c.read_i8());
        case VT_UINT16:
            return static_cast<std::int64_t>(c.read_u16());
        case VT_INT16:
            return static_cast<std::int64_t>(c.read_i16());
        case VT_UINT32:
            return static_cast<std::int64_t>(c.read_u32());
        case VT_INT32:
            return static_cast<std::int64_t>(c.read_i32());
        case VT_UINT64:
            return static_cast<std::int64_t>(c.read_u64());
        case VT_INT64:
            return static_cast<std::int64_t>(c.read_i64());
        case VT_FLOAT32:
            return static_cast<double>(c.read_f32());
        case VT_FLOAT64:
            return c.read_f64();
        case VT_BOOL:
            return c.read_u8() != 0;
        case VT_STRING:
            return c.read_string();
        default:
            throw std::runtime_error("gguf: unknown metadata scalar value type");
    }
}

// Read an ARRAY value: { elem_type: u32; len: u64; then len elements }. We keep a
// flat vector typed by the element kind (ints -> int64, floats -> double, strings
// -> string), matching the three vector alternatives of MetadataValue. Arrays of
// arrays are not allowed by the spec, so a nested ARRAY elem_type is an error.
MetadataValue read_array(Cursor& c) {
    std::uint32_t elem_type = c.read_u32();
    std::uint64_t len       = c.read_u64();

    switch (elem_type) {
        case VT_UINT8:
        case VT_INT8:
        case VT_UINT16:
        case VT_INT16:
        case VT_UINT32:
        case VT_INT32:
        case VT_UINT64:
        case VT_INT64: {
            std::vector<std::int64_t> out;
            out.reserve(len);
            for (std::uint64_t i = 0; i < len; ++i) {
                out.push_back(std::get<std::int64_t>(read_scalar(c, elem_type)));
            }
            return out;
        }
        case VT_FLOAT32:
        case VT_FLOAT64: {
            std::vector<double> out;
            out.reserve(len);
            for (std::uint64_t i = 0; i < len; ++i) {
                out.push_back(std::get<double>(read_scalar(c, elem_type)));
            }
            return out;
        }
        case VT_BOOL: {
            // No bool-vector alternative; widen to int64 like the other ints.
            std::vector<std::int64_t> out;
            out.reserve(len);
            for (std::uint64_t i = 0; i < len; ++i) {
                out.push_back(c.read_u8() != 0 ? 1 : 0);
            }
            return out;
        }
        case VT_STRING: {
            // Tokenizer vocab arrives this way -- the one array kind we must support.
            std::vector<std::string> out;
            out.reserve(len);
            for (std::uint64_t i = 0; i < len; ++i) {
                out.push_back(c.read_string());
            }
            return out;
        }
        default:
            throw std::runtime_error("gguf: invalid array element type (arrays cannot nest)");
    }
}

}  // namespace

// --- Typed accessors --------------------------------------------------------
// Each looks the key up, then pulls the matching variant alternative. A missing
// key or a type mismatch throws, so a malformed config fails loudly at the call
// site rather than silently returning a default.

std::int64_t GGUF::i64(const std::string& key) const {
    auto it = metadata.find(key);
    if (it == metadata.end()) {
        throw std::runtime_error("gguf: missing metadata key: " + key);
    }
    if (auto* p = std::get_if<std::int64_t>(&it->second)) {
        return *p;
    }
    throw std::runtime_error("gguf: metadata key is not an integer: " + key);
}

double GGUF::f64(const std::string& key) const {
    auto it = metadata.find(key);
    if (it == metadata.end()) {
        throw std::runtime_error("gguf: missing metadata key: " + key);
    }
    if (auto* p = std::get_if<double>(&it->second)) {
        return *p;
    }
    throw std::runtime_error("gguf: metadata key is not a float: " + key);
}

bool GGUF::boolean(const std::string& key) const {
    auto it = metadata.find(key);
    if (it == metadata.end()) {
        throw std::runtime_error("gguf: missing metadata key: " + key);
    }
    if (auto* p = std::get_if<bool>(&it->second)) {
        return *p;
    }
    throw std::runtime_error("gguf: metadata key is not a bool: " + key);
}

const std::string& GGUF::str(const std::string& key) const {
    auto it = metadata.find(key);
    if (it == metadata.end()) {
        throw std::runtime_error("gguf: missing metadata key: " + key);
    }
    if (auto* p = std::get_if<std::string>(&it->second)) {
        return *p;
    }
    throw std::runtime_error("gguf: metadata key is not a string: " + key);
}

const std::vector<std::string>& GGUF::str_array(const std::string& key) const {
    auto it = metadata.find(key);
    if (it == metadata.end()) {
        throw std::runtime_error("gguf: missing metadata key: " + key);
    }
    if (auto* p = std::get_if<std::vector<std::string>>(&it->second)) {
        return *p;
    }
    throw std::runtime_error("gguf: metadata key is not a string array: " + key);
}

const std::vector<double>& GGUF::f64_array(const std::string& key) const {
    auto it = metadata.find(key);
    if (it == metadata.end()) {
        throw std::runtime_error("gguf: missing metadata key: " + key);
    }
    if (auto* p = std::get_if<std::vector<double>>(&it->second)) {
        return *p;
    }
    throw std::runtime_error("gguf: metadata key is not a float array: " + key);
}

const std::vector<std::int64_t>& GGUF::i64_array(const std::string& key) const {
    auto it = metadata.find(key);
    if (it == metadata.end()) {
        throw std::runtime_error("gguf: missing metadata key: " + key);
    }
    if (auto* p = std::get_if<std::vector<std::int64_t>>(&it->second)) {
        return *p;
    }
    throw std::runtime_error("gguf: metadata key is not an integer array: " + key);
}

const Tensor& GGUF::tensor(const std::string& name) const {
    auto it = tensors.find(name);
    if (it == tensors.end()) {
        throw std::runtime_error("gguf: missing tensor: " + name);
    }
    return it->second;
}

// --- Loader -----------------------------------------------------------------

GGUF load_gguf(const std::string& path) {
    GGUF g;
    // Hold the mapping in a shared_ptr so every tensor view can co-own it; the
    // mmap then lives exactly as long as the GGUF (and any view it hands out).
    g.mapping = std::make_shared<MappedFile>(path);

    const std::byte* base = g.mapping->data();
    const std::byte* end  = base + g.mapping->size();
    Cursor           c(base, end);

    // HEADER: magic, version, tensor_count, metadata_kv_count (24 bytes).
    std::uint32_t magic = c.read_u32();
    if (magic != GGUF_MAGIC) {
        throw std::runtime_error("gguf: bad magic (not a GGUF file): " + path);
    }
    g.version = c.read_u32();
    if (g.version != 2 && g.version != 3) {
        throw std::runtime_error("gguf: unsupported version " + std::to_string(g.version) +
                                 " (only 2 and 3 are supported)");
    }
    std::uint64_t tensor_count      = c.read_u64();
    std::uint64_t metadata_kv_count = c.read_u64();

    // METADATA: metadata_kv_count entries of { key: string; value_type: u32; value }.
    g.metadata.reserve(metadata_kv_count);
    for (std::uint64_t i = 0; i < metadata_kv_count; ++i) {
        std::string   key  = c.read_string();
        std::uint32_t type = c.read_u32();
        if (type == VT_ARRAY) {
            g.metadata.emplace(std::move(key), read_array(c));
        } else {
            g.metadata.emplace(std::move(key), read_scalar(c, type));
        }
    }

    // Data alignment: the "general.alignment" u32 kv overrides the default of 32.
    std::uint64_t alignment = DEFAULT_ALIGNMENT;
    if (g.has("general.alignment")) {
        alignment = g.u32("general.alignment");
        if (alignment == 0) {
            throw std::runtime_error("gguf: general.alignment is zero");
        }
    }

    // TENSOR DIRECTORY: tensor_count entries of
    // { name: string; n_dims: u32; dims: u64[n_dims]; type: u32; offset: u64 }.
    // We collect the raw entries first, then resolve them against the data section
    // (whose start we only know once the directory has been fully walked).
    struct TensorInfo {
        std::string               name;
        std::vector<std::int64_t> dims;    // ggml ne[], innermost-first
        std::uint32_t             type;
        std::uint64_t             offset;  // relative to data_start
    };
    std::vector<TensorInfo> infos;
    infos.reserve(tensor_count);

    for (std::uint64_t i = 0; i < tensor_count; ++i) {
        TensorInfo   info;
        info.name           = c.read_string();
        std::uint32_t n_dims = c.read_u32();
        info.dims.reserve(n_dims);
        for (std::uint32_t d = 0; d < n_dims; ++d) {
            info.dims.push_back(static_cast<std::int64_t>(c.read_u64()));
        }
        info.type   = c.read_u32();
        info.offset = c.read_u64();
        infos.push_back(std::move(info));
    }

    // DATA SECTION starts at align_up(byte offset just past the directory). Each
    // tensor's bytes are at data_start + offset (offset is a multiple of alignment).
    std::uint64_t    dir_end_off = static_cast<std::uint64_t>(c.pos() - base);
    std::uint64_t    data_off    = align_up(dir_end_off, alignment);
    const std::byte* data_start  = base + data_off;
    if (data_off > g.mapping->size()) {
        throw std::runtime_error("gguf: data section starts past end of file");
    }

    g.tensors.reserve(tensor_count);
    for (auto& info : infos) {
        // We only support F32 today -- abort clearly on anything else.
        if (info.type != GGML_TYPE_F32) {
            throw std::runtime_error("gguf: unsupported tensor type " + std::to_string(info.type) +
                                     " for '" + info.name + "' (only F32 is supported)");
        }

        // ggml stores dims as ne[] with ne[0] = innermost/fastest dim. Our Tensor is
        // row-major with shape[last] innermost, so REVERSE ne[] to get our shape:
        // an HF weight (out, in) stored as ne=[in, out] becomes shape {out, in}.
        std::vector<int> shape;
        shape.reserve(info.dims.size());
        std::int64_t     num_elements = 1;
        for (auto it = info.dims.rbegin(); it != info.dims.rend(); ++it) {
            shape.push_back(static_cast<int>(*it));
            num_elements *= *it;
        }

        // Bounds-check the whole tensor against the mapping before making a view.
        std::uint64_t need = data_off + info.offset + static_cast<std::uint64_t>(num_elements) * sizeof(float);
        if (need > g.mapping->size()) {
            throw std::runtime_error("gguf: tensor '" + info.name + "' data runs past end of file");
        }

        // The data section is 4-byte F32 and (per spec) aligned, so reinterpreting
        // the bytes as float is safe here. The view shares the mapping as its owner_.
        const float* tensor_data = reinterpret_cast<const float*>(data_start + info.offset);
        g.tensors.emplace(info.name, Tensor(std::move(shape), tensor_data, g.mapping));
    }

    return g;
}

}  // namespace gguf
