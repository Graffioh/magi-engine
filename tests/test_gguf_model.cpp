#include "gguf.h"
#include "gguf_model.h"
#include "mmap.h"
#include "model.h"
#include "model_config.h"
#include "tensor.h"
#include "test_utils.h"

#include <cassert>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

// GGUF -> Model adapter tests. The real test is a round-trip: tools/gen_toy_gguf.py
// writes tests/gguf/toy.gguf carrying the EXACT toy weights (imported from
// gen_golden_weights.py) under canonical ggml names + llama.* metadata. We load that
// GGUF through the C++ loader, build a Model via build_model_from_gguf, and check its
// forward/generate against the SAME goldens test_golden.cpp uses -- so a green run
// proves the adapter wires the real names/keys to the right layers.
//
// Both fixtures are gitignored (regenerate with the two python tools), so we SKIP
// rather than fail when toy.gguf is absent -- same posture as the other suites.
// MAGI_GGUF_DIR / MAGI_GOLDEN_DIR are injected by CMake (CWD-independent).

namespace {
namespace fs = std::filesystem;

const fs::path GGUF_DIR   = MAGI_GGUF_DIR;
const fs::path GOLDEN_DIR = MAGI_GOLDEN_DIR;

// Map a `<name>.f32` golden blob and return a read-only Tensor view over it with
// `shape`. Same approach as load_golden in test_golden.cpp: the mmap is kept alive
// by the Tensor's owner_ for as long as the returned Tensor lives.
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

// Toy dims -- must match tools/gen_golden_weights.py (the GGUF carries the same).
constexpr int T              = 3;  // len(INPUT_IDS)
constexpr int VOCAB          = 32;
constexpr int HIDDEN         = 8;
constexpr int INTERMEDIATE   = 16;
constexpr int N_LAYERS       = 2;
constexpr int N_HEADS        = 2;
constexpr int N_KV_HEADS     = 1;
constexpr int HEAD_DIM       = 4;
constexpr int MAX_SEQ        = 16;
constexpr int MAX_NEW_TOKENS = 8;

}  // namespace

void run_gguf_model_tests(TestState& s) {
    if (!fs::exists(GGUF_DIR / "toy.gguf")) {
        std::cout << "  [SKIP] gguf-model tests -- run: python3 tools/gen_toy_gguf.py\n";
        return;
    }

    const std::vector<int> ids = { 5, 0, 11 };  // must match INPUT_IDS in gen_golden_weights.py

    // --- config_from_gguf reads the llama.* metadata into a ModelConfig ---
    // Load once, read the config off the still-live GGUF (build_model consumes a move).
    {
        gguf::GGUF  g   = gguf::load_gguf((GGUF_DIR / "toy.gguf").string());
        ModelConfig cfg = config_from_gguf(g);

        check(s, "gguf-model: cfg vocab_size", std::vector<int>{ cfg.vocab_size }, std::vector<int>{ VOCAB });
        check(s, "gguf-model: cfg hidden_size", std::vector<int>{ cfg.hidden_size }, std::vector<int>{ HIDDEN });
        check(s, "gguf-model: cfg intermediate_size", std::vector<int>{ cfg.intermediate_size },
              std::vector<int>{ INTERMEDIATE });
        check(s, "gguf-model: cfg n_layers", std::vector<int>{ cfg.n_layers }, std::vector<int>{ N_LAYERS });
        check(s, "gguf-model: cfg n_heads", std::vector<int>{ cfg.n_heads }, std::vector<int>{ N_HEADS });
        check(s, "gguf-model: cfg n_kv_heads", std::vector<int>{ cfg.n_kv_heads }, std::vector<int>{ N_KV_HEADS });
        check(s, "gguf-model: cfg head_dim", std::vector<int>{ cfg.head_dim }, std::vector<int>{ HEAD_DIM });
        check(s, "gguf-model: cfg max_seq_len", std::vector<int>{ cfg.max_seq_len }, std::vector<int>{ MAX_SEQ });

        // rms_eps is a float; compare with a tolerance via the Tensor check().
        Tensor eps_got({ 1 });
        eps_got.data_ptr()[0] = cfg.rms_eps;
        check(s, "gguf-model: cfg rms_eps", eps_got, { 1e-5f }, { 1 }, 1e-7f);
    }

    // --- build_model_from_gguf -> forward vs golden.logits ---
    // Loosest tolerance (1e-4): fp32 matmul/softmax drift accumulates across both
    // layers, same as the full-model golden in test_golden.cpp.
    {
        gguf::GGUF g = gguf::load_gguf((GGUF_DIR / "toy.gguf").string());
        Model      m = build_model_from_gguf(std::move(g));

        Tensor logits({ T, VOCAB });
        m.forward(ids, logits);

        Tensor golden = load_golden({ T, VOCAB }, "golden.logits");
        check(s, "gguf-model: full model logits", logits, flat(golden), { T, VOCAB }, 1e-4f);
    }

    // --- build_model_from_gguf -> greedy generate vs golden.gen_ids (exact) ---
    // Deterministic greedy decode, so the produced id sequence must match exactly.
    {
        gguf::GGUF g = gguf::load_gguf((GGUF_DIR / "toy.gguf").string());
        Model      m = build_model_from_gguf(std::move(g));

        std::vector<int> out_ids = m.generate(ids, MAX_NEW_TOKENS);

        Tensor           gen = load_golden({ T + MAX_NEW_TOKENS }, "golden.gen_ids");
        std::vector<int> expected_ids;
        for (int i = 0; i < gen.num_elements(); ++i) {
            expected_ids.push_back(static_cast<int>(gen.data_ptr()[i]));
        }
        check(s, "gguf-model: greedy generation", out_ids, expected_ids);
    }
}
