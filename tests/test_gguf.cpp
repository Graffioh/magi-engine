#include "gguf.h"
#include "mmap.h"
#include "tensor.h"
#include "test_utils.h"

#include <cassert>
#include <filesystem>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

// GGUF loader round-trip tests. tools/gen_tiny_gguf.py writes tests/gguf/tiny.gguf
// (preferably via the canonical gguf writer, so this validates our reader against
// the real format) plus the *source* arrays as raw float32 blobs. We load the GGUF
// and check that the parsed metadata and the mmap tensor views match exactly what
// the Python tool wrote.
//
// The fixture is gitignored (regenerate with `python3 tools/gen_tiny_gguf.py`), so
// if it is absent we SKIP rather than fail the suite -- same posture as the golden
// tests. MAGI_GGUF_DIR is injected by CMake so the path is CWD-independent.

namespace {
namespace fs = std::filesystem;

const fs::path GGUF_DIR = MAGI_GGUF_DIR;

// Map a `<name>.f32` source blob and return a read-only Tensor view over it with
// `shape`. Identical in spirit to load_golden in test_golden.cpp: the mmap is kept
// alive by the Tensor's owner_ (shared_ptr), so the view is valid as long as the
// returned Tensor lives.
Tensor load_blob(const std::vector<int>& shape, const std::string& name) {
    auto         mapping = std::make_shared<MappedFile>((GGUF_DIR / (name + ".f32")).string());
    const float* base    = reinterpret_cast<const float*>(mapping->data());

    int elements = 1;
    for (int d : shape) {
        elements *= d;
    }
    assert(mapping->size() == elements * sizeof(float) && "gguf source blob size does not match shape");

    return Tensor(shape, base, std::move(mapping));
}

// Flatten a Tensor's contents into a vector<float> for check().
std::vector<float> flat(const Tensor& t) {
    const float* p = t.data_ptr();
    return std::vector<float>(p, p + t.num_elements());
}

}  // namespace

void run_gguf_tests(TestState& s) {
    if (!fs::exists(GGUF_DIR / "tiny.gguf")) {
        std::cout << "  [SKIP] gguf tests -- run: python3 tools/gen_tiny_gguf.py\n";
        return;
    }

    gguf::GGUF model = gguf::load_gguf((GGUF_DIR / "tiny.gguf").string());

    // --- header: version + tensor/metadata counts (must match the Python tool) ---
    {
        bool version_ok = (model.version == 2 || model.version == 3);
        check(s, "gguf: version is 2 or 3", std::vector<int>{ version_ok ? 1 : 0 }, std::vector<int>{ 1 });
        // tiny.gguf carries 2 tensors and 6 metadata kv entries (u32, f32, bool,
        // general.architecture, general.alignment, and the tokens string array).
        check(s, "gguf: tensor count", std::vector<int>{ (int) model.tensors.size() }, std::vector<int>{ 2 });
        check(s, "gguf: metadata count", std::vector<int>{ (int) model.metadata.size() }, std::vector<int>{ 6 });
    }

    // --- metadata round-trips across every KV type the parser handles ---
    {
        // u32: stored widened to int64, read back via u32().
        check(s, "gguf: meta u32", std::vector<int>{ (int) model.u32("magi.test.u32") }, std::vector<int>{ 1234 });

        // f32: stored widened to double, read back via f32() (tolerance for the cast).
        Tensor f32_got({ 1 });
        f32_got.data_ptr()[0] = model.f32("magi.test.f32");
        check(s, "gguf: meta f32", f32_got, { 2.5f }, { 1 }, 1e-6f);

        // bool.
        check(s, "gguf: meta bool", std::vector<int>{ model.boolean("magi.test.bool") ? 1 : 0 },
              std::vector<int>{ 1 });

        // string: general.architecture == "llama".
        check(s, "gguf: meta string", std::vector<int>{ model.str("general.architecture") == "llama" ? 1 : 0 },
              std::vector<int>{ 1 });

        // string array: size + a couple of elements.
        const std::vector<std::string>& tokens = model.str_array("magi.test.tokens");
        check(s, "gguf: meta str-array size", std::vector<int>{ (int) tokens.size() }, std::vector<int>{ 4 });
        bool tokens_ok =
            tokens.size() == 4 && tokens[0] == "<unk>" && tokens[1] == "<s>" && tokens[3] == "hi";
        check(s, "gguf: meta str-array values", std::vector<int>{ tokens_ok ? 1 : 0 }, std::vector<int>{ 1 });
    }

    // --- tensors: reversed-ne[] shape + exact float values vs the source blobs ---
    // numpy wrote (4, 3); ggml stores ne=[3, 4]; our reader reverses it back to
    // {4, 3}. The view's bytes must equal the .f32 blob byte-for-byte (same mmap'd
    // source values), so we compare element-wise with a tight tolerance.
    {
        const Tensor& q = model.tensor("blk.0.attn_q.weight");
        Tensor        q_src = load_blob({ 4, 3 }, "blk.0.attn_q.weight");
        check(s, "gguf: tensor blk.0.attn_q.weight", q, flat(q_src), { 4, 3 }, 1e-6f);

        const Tensor& n = model.tensor("output_norm.weight");
        Tensor        n_src = load_blob({ 5 }, "output_norm.weight");
        check(s, "gguf: tensor output_norm.weight", n, flat(n_src), { 5 }, 1e-6f);
    }
}
