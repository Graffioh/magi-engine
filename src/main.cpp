#include "tensor.h"
#include "ops.h"

#include <cmath>
#include <iostream>
#include <string>
#include <vector>

static void print_shape(const Tensor& t) {
    std::cout << "  shape   = [";
    for (int i = 0; i < t.get_rank(); ++i) {
        std::cout << t.get_shape()[i];
        if (i + 1 < t.get_rank()) std::cout << ", ";
    }
    std::cout << "]\n";

    std::cout << "  strides = [";
    for (int i = 0; i < t.get_rank(); ++i) {
        std::cout << t.get_strides()[i];
        if (i + 1 < t.get_rank()) std::cout << ", ";
    }
    std::cout << "]\n";
}

static void fill(Tensor& t, const std::vector<float>& vals) {
    float* p = t.data_ptr();
    for (size_t i = 0; i < vals.size(); ++i) p[i] = vals[i];
}

int main() {
    int passed = 0, failed = 0;

    auto check = [&](const std::string& name, const Tensor& got, const std::vector<float>& expected, float tol = 1e-5f) {
        const auto& shape = got.get_shape();
        int total = 1;
        for (int d : shape) total *= d;
        if ((int)expected.size() != total) {
            std::cout << "  [FAIL] " << name << ": expected size " << expected.size()
                      << " but tensor has " << total << " elements\n";
            failed++; return;
        }
        bool ok = true;
        for (int i = 0; i < total; ++i) {
            if (std::fabs(got.data_ptr()[i] - expected[i]) > tol) { ok = false; break; }
        }
        if (ok) { std::cout << "  [PASS] " << name << "\n"; passed++; }
        else {
            std::cout << "  [FAIL] " << name << "\n    got:      [";
            for (int i = 0; i < total; ++i) std::cout << got.data_ptr()[i] << (i+1<total?", ":"");
            std::cout << "]\n    expected: [";
            for (int i = 0; i < total; ++i) std::cout << expected[i] << (i+1<total?", ":"");
            std::cout << "]\n";
            failed++;
        }
    };

    std::cout << "=== Tensor basics ===\n";
    Tensor t({2, 3, 4});
    print_shape(t);
    std::cout << "\n";

    // ---------------------------------------------------------------
    // matmul tests
    // ---------------------------------------------------------------
    std::cout << "=== matmul: 2D (2,2) x (2,2) ===\n";
    {
        Tensor A({2, 2}), B({2, 2}), C({2, 2});
        fill(A, {1, 2, 3, 4});
        fill(B, {5, 6, 7, 8});
        ops::matmul(A, B, C);
        check("(2,2) x (2,2)", C, {19, 22, 43, 50});
    }

    std::cout << "=== matmul: 2D (2,3) x (3,2) ===\n";
    {
        Tensor A({2, 3}), B({3, 2}), C({2, 2});
        fill(A, {1, 2, 3, 4, 5, 6});
        fill(B, {1, 2, 3, 4, 5, 6});
        ops::matmul(A, B, C);
        check("(2,3) x (3,2)", C, {22, 28, 49, 64});
    }

    std::cout << "=== matmul: 2D identity (3,3) x (3,3) ===\n";
    {
        Tensor I({3, 3}), B({3, 3}), C({3, 3});
        fill(I, {1, 0, 0, 0, 1, 0, 0, 0, 1});
        fill(B, {7, 8, 9, 1, 2, 3, 4, 5, 6});
        ops::matmul(I, B, C);
        check("identity passes B through", C, {7, 8, 9, 1, 2, 3, 4, 5, 6});
    }

    std::cout << "=== matmul: 3D batched (2,2,2) x (2,2,2) ===\n";
    {
        Tensor A({2, 2, 2}), B({2, 2, 2}), C({2, 2, 2});
        fill(A, {1, 2, 3, 4,  1, 0, 0, 1});
        fill(B, {5, 6, 7, 8,  9, 8, 7, 6});
        ops::matmul(A, B, C);
        check("batched (2,2,2) x (2,2,2)", C, {19, 22, 43, 50,  9, 8, 7, 6});
    }

    // ---------------------------------------------------------------
    // RMSNorm tests
    // ---------------------------------------------------------------
    std::cout << "\n=== rmsnorm: 1 row [3, 4], W=[1,1] ===\n";
    {
        // RMS = sqrt((9+16)/2) = sqrt(12.5) ≈ 3.5355339
        // out ≈ [3/3.5355, 4/3.5355] ≈ [0.84853, 1.13137]
        Tensor X({1, 2}), W({2}), Y({1, 2});
        fill(X, {3, 4});
        fill(W, {1, 1});
        ops::RMSNorm(X, W, Y, 0.0f);
        check("[3,4] W=[1,1]", Y, {0.84852813f, 1.13137085f}, 1e-4f);
    }

    std::cout << "=== rmsnorm: 1 row [3, 4], W=[2, 0.5] ===\n";
    {
        // base = [0.84853, 1.13137]; W = [2, 0.5] → [1.69706, 0.56569]
        Tensor X({1, 2}), W({2}), Y({1, 2});
        fill(X, {3, 4});
        fill(W, {2.0f, 0.5f});
        ops::RMSNorm(X, W, Y, 0.0f);
        check("[3,4] W=[2,0.5]", Y, {1.69705627f, 0.56568542f}, 1e-4f);
    }

    std::cout << "=== rmsnorm: 2 rows, each independent ===\n";
    {
        // Row 0: [3, 4]    → RMS=sqrt(12.5) → [0.84853, 1.13137]
        // Row 1: [1, 1]    → RMS=1          → [1.0,     1.0]
        Tensor X({2, 2}), W({2}), Y({2, 2});
        fill(X, {3, 4,  1, 1});
        fill(W, {1, 1});
        ops::RMSNorm(X, W, Y, 0.0f);
        check("two rows normalized independently", Y, {0.84852813f, 1.13137085f, 1.0f, 1.0f}, 1e-4f);
    }

    std::cout << "=== rmsnorm: 3D (batch, seq, hidden) ===\n";
    {
        // Shape (2, 2, 2): 4 rows of length 2.
        // All four rows are [3, 4] → RMS = sqrt(12.5) → [0.84853, 1.13137]
        Tensor X({2, 2, 2}), W({2}), Y({2, 2, 2});
        fill(X, {3, 4, 3, 4, 3, 4, 3, 4});
        fill(W, {1, 1});
        ops::RMSNorm(X, W, Y, 0.0f);
        std::vector<float> expected;
        for (int r = 0; r < 4; ++r) {
            expected.push_back(0.84852813f);
            expected.push_back(1.13137085f);
        }
        check("3D shape (2,2,2) — 4 independent rows", Y, expected, 1e-4f);
    }

    std::cout << "=== rmsnorm: eps prevents divide-by-zero on all-zero row ===\n";
    {
        Tensor X({1, 3}), W({3}), Y({1, 3});
        fill(X, {0, 0, 0});
        fill(W, {1, 1, 1});
        ops::RMSNorm(X, W, Y, 1e-5f);
        // sum_sq = 0, rms = sqrt(0/3 + 1e-5) ≈ 0.003162; inv_rms ≈ 316.23
        // each output = 0 * inv_rms * 1 = 0 → no NaN, just zeros
        check("all-zero row stays finite", Y, {0.0f, 0.0f, 0.0f}, 1e-4f);
    }

    std::cout << "\nresults: " << passed << " passed, " << failed << " failed\n";
    return failed == 0 ? 0 : 1;
}
