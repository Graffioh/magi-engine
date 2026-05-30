#include "layers.h"
#include "mmap.h"
#include "ops.h"
#include "tensor.h"
#include "test_utils.h"

#include <cassert>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

// Golden-reference tests. These load the raw float32 blobs produced by
// tools/gen_golden_weights.py and check the engine's ops against the NumPy reference.
// The blobs are gitignored (regenerate with `python3 tools/gen_golden_weights.py`), so
// if they are absent we SKIP rather than fail the whole suite.
//
// MAGI_GOLDEN_DIR is injected by CMake so the path does not depend on the
// current working directory.

namespace {
namespace fs = std::filesystem;

const fs::path GOLDEN_DIR = MAGI_GOLDEN_DIR;

// Map a `<name>.f32` blob and return a Tensor view over it with `shape`.
// The mmap is kept alive by the Tensor's owner_ (shared_ptr), so the view is
// valid for as long as the returned Tensor lives. Read-only by construction.
Tensor load_golden(const std::vector<int>& shape, const std::string& name) {
    auto         mapping = std::make_shared<MappedFile>((GOLDEN_DIR / (name + ".f32")).string());
    const float* base    = reinterpret_cast<const float*>(mapping->data());

    int elements = 1;
    for (int d : shape) {
        elements *= d;
    }
    assert(mapping->size() == elements * sizeof(float) && "golden blob size does not match shape");

    return Tensor(shape, base, std::move(mapping));
}

// Flatten a Tensor's contents into a vector<float> for check().
std::vector<float> flat(const Tensor& t) {
    const float* p = t.data_ptr();
    return std::vector<float>(p, p + t.num_elements());
}

// Copy a `<name>.f32` blob into a mutable heap Tensor. Needed for in-place ops
// (RoPE) whose input must be writable -- load_golden returns a read-only mmap.
Tensor load_golden_mut(const std::vector<int>& shape, const std::string& name) {
    Tensor src = load_golden(shape, name);
    Tensor dst(shape);
    std::memcpy(dst.data_ptr(), src.data_ptr(), src.num_elements() * sizeof(float));
    return dst;
}

// Toy dims -- must match tools/gen_golden_weights.py.
constexpr int T            = 3;  // len(INPUT_IDS)
constexpr int VOCAB        = 32;
constexpr int HIDDEN       = 8;
constexpr int INTERMEDIATE = 16;
constexpr int HEAD_DIM     = 4;
constexpr int N_HEADS      = 2;
constexpr int N_KV_HEADS   = 1;

}  // namespace

void run_golden_tests(TestState& s) {
    if (!fs::exists(GOLDEN_DIR / "golden.embed_out.f32")) {
        std::cout << "  [SKIP] golden tests -- run: python3 tools/gen_golden_weights.py\n";
        return;
    }

    const std::vector<int> ids = { 5, 0, 11 };  // must match INPUT_IDS in gen_golden_weights.py

    // --- embedding lookup vs golden (exact byte copy -> tight tolerance) ---
    {
        Tensor We = load_golden({ VOCAB, HIDDEN }, "embed.weight");
        Tensor out({ T, HIDDEN });
        ops::embed(ids, We, out);
        Tensor golden = load_golden({ T, HIDDEN }, "golden.embed_out");
        check(s, "golden: embed", out, flat(golden), { T, HIDDEN }, 1e-6f);
    }

    // --- RMSNorm vs golden (fed the golden embed_out, so it tests RMSNorm alone) ---
    {
        Tensor x = load_golden({ T, HIDDEN }, "golden.embed_out");
        Tensor w = load_golden({ HIDDEN }, "layer0.attn_norm.weight");
        Tensor out({ T, HIDDEN });
        ops::RMSNorm(x, w, out);  // default eps 1e-5, matching gen_golden_weights.py
        Tensor golden = load_golden({ T, HIDDEN }, "golden.rmsnorm_out");
        check(s, "golden: rmsnorm", out, flat(golden), { T, HIDDEN }, 1e-5f);
    }

    // --- SwiGLU MLP vs golden (matmul-heavy -> looser tolerance for fp32 drift) ---
    {
        Tensor x  = load_golden({ T, HIDDEN }, "golden.embed_out");
        Tensor wg = load_golden({ INTERMEDIATE, HIDDEN }, "layer0.mlp.gate.weight");
        Tensor wu = load_golden({ INTERMEDIATE, HIDDEN }, "layer0.mlp.up.weight");
        Tensor wd = load_golden({ HIDDEN, INTERMEDIATE }, "layer0.mlp.down.weight");
        MLP    mlp(LinearLayer(std::move(wg)), LinearLayer(std::move(wu)), LinearLayer(std::move(wd)));
        Tensor out({ T, HIDDEN });
        mlp.forward(x, out);
        Tensor golden = load_golden({ T, HIDDEN }, "golden.mlp_out");
        check(s, "golden: mlp", out, flat(golden), { T, HIDDEN }, 1e-4f);
    }

    // --- RoPE vs golden (loads the saved Q/K inputs, so it tests RoPE alone) ---
    // Tolerance is loosened to 1e-4 because float32 sin/cos/pow differ by a few
    // ULP between C libm and NumPy, and those errors accumulate per pair.
    {
        Tensor Q = load_golden_mut({ T, N_HEADS * HEAD_DIM }, "golden.rope_q_in");
        Tensor K = load_golden_mut({ T, N_KV_HEADS * HEAD_DIM }, "golden.rope_k_in");
        ops::RoPE(Q, K, HEAD_DIM);  // default start_pos=0, base=10000
        Tensor golden_q = load_golden({ T, N_HEADS * HEAD_DIM }, "golden.rope_q_out");
        Tensor golden_k = load_golden({ T, N_KV_HEADS * HEAD_DIM }, "golden.rope_k_out");
        check(s, "golden: rope (Q)", Q, flat(golden_q), { T, N_HEADS * HEAD_DIM }, 1e-4f);
        check(s, "golden: rope (K)", K, flat(golden_k), { T, N_KV_HEADS * HEAD_DIM }, 1e-4f);
    }
}
