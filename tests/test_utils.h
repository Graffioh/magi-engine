#pragma once

#include "tensor.h"

#include <cmath>
#include <iostream>
#include <string>
#include <vector>

// Accumulates pass/fail counts across every run_<component>_tests call.
// tests/test_main.cpp prints the totals at the end and uses `failed` as the
// process exit code, so a non-zero exit means at least one assertion failed.
struct TestState {
    int passed = 0;
    int failed = 0;
};

// Writes `vals` into the tensor's flat storage in row-major order. Useful
// for hand-seeding small tensors in tests, e.g. fill(A, {1, 2, 3, 4}) for
// a (2, 2) tensor. `vals.size()` must equal the total element count; this
// helper does not check.
inline void fill(Tensor& t, const std::vector<float>& vals) {
    float* p = t.data_ptr();
    for (size_t i = 0; i < vals.size(); ++i) p[i] = vals[i];
}

// Compares `got` against `expected` elementwise with float tolerance `tol`
// and prints a [PASS]/[FAIL] line, updating `s` accordingly. `expected` is
// the flat row-major contents of `got`; the element count must match the
// product of `got`'s shape. On failure, both arrays are printed for diff.
//
// If `expected_shape` is non-empty, the tensor's shape is checked first;
// a mismatch fails immediately without comparing data (indices would be
// meaningless once layout disagrees). Pass `{}` to skip the shape check.
inline void check(TestState& s, const std::string& name, const Tensor& got,
                  const std::vector<float>& expected,
                  const std::vector<int>& expected_shape = {},
                  float tol = 1e-5f) {
    const auto& shape = got.shape();
    if (!expected_shape.empty()) {
        bool shape_ok = shape.size() == expected_shape.size();
        for (size_t i = 0; shape_ok && i < shape.size(); ++i) {
            if (shape[i] != expected_shape[i]) shape_ok = false;
        }
        if (!shape_ok) {
            std::cout << "  [FAIL] " << name << " (shape)\n    expected shape: [";
            for (size_t i = 0; i < expected_shape.size(); ++i)
                std::cout << expected_shape[i] << (i + 1 < expected_shape.size() ? ", " : "");
            std::cout << "]\n    got shape:      [";
            for (size_t i = 0; i < shape.size(); ++i)
                std::cout << shape[i] << (i + 1 < shape.size() ? ", " : "");
            std::cout << "]\n";
            s.failed++;
            return;
        }
    }
    int total = 1;
    for (int d : shape) total *= d;
    if ((int)expected.size() != total) {
        std::cout << "  [FAIL] " << name << ": expected size " << expected.size()
                  << " but tensor has " << total << " elements\n";
        s.failed++;
        return;
    }
    bool ok = true;
    for (int i = 0; i < total; ++i) {
        if (std::fabs(got.data_ptr()[i] - expected[i]) > tol) {
            ok = false;
            break;
        }
    }
    if (ok) {
        std::cout << "  [PASS] " << name;
        if (!expected_shape.empty()) {
            std::cout << "  shape=[";
            for (size_t i = 0; i < shape.size(); ++i)
                std::cout << shape[i] << (i + 1 < shape.size() ? ", " : "");
            std::cout << "]";
        }
        std::cout << "  expected=[";
        for (int i = 0; i < total; ++i) std::cout << expected[i] << (i + 1 < total ? ", " : "");
        std::cout << "]  got=[";
        for (int i = 0; i < total; ++i) std::cout << got.data_ptr()[i] << (i + 1 < total ? ", " : "");
        std::cout << "]\n";
        s.passed++;
    } else {
        std::cout << "  [FAIL] " << name;
        if (!expected_shape.empty()) {
            std::cout << "  shape=[";
            for (size_t i = 0; i < shape.size(); ++i)
                std::cout << shape[i] << (i + 1 < shape.size() ? ", " : "");
            std::cout << "]";
        }
        std::cout << "\n    expected: [";
        for (int i = 0; i < total; ++i) std::cout << expected[i] << (i + 1 < total ? ", " : "");
        std::cout << "]\n    got:      [";
        for (int i = 0; i < total; ++i) std::cout << got.data_ptr()[i] << (i + 1 < total ? ", " : "");
        std::cout << "]\n";
        s.failed++;
    }
}

// Compares two integer vectors for exact equality. Used for metadata checks
// such as shape and strides, where no float tolerance applies. Same
// [PASS]/[FAIL] output style as the Tensor overload above.
inline void check(TestState& s, const std::string& name,
                  const std::vector<int>& got,
                  const std::vector<int>& expected) {
    bool ok = got.size() == expected.size();
    for (size_t i = 0; ok && i < got.size(); ++i) {
        if (got[i] != expected[i]) ok = false;
    }
    if (ok) {
        std::cout << "  [PASS] " << name << "  expected=[";
        for (size_t i = 0; i < expected.size(); ++i) std::cout << expected[i] << (i + 1 < expected.size() ? ", " : "");
        std::cout << "]  got=[";
        for (size_t i = 0; i < got.size(); ++i) std::cout << got[i] << (i + 1 < got.size() ? ", " : "");
        std::cout << "]\n";
        s.passed++;
    } else {
        std::cout << "  [FAIL] " << name << "\n    expected: [";
        for (size_t i = 0; i < expected.size(); ++i) std::cout << expected[i] << (i + 1 < expected.size() ? ", " : "");
        std::cout << "]\n    got:      [";
        for (size_t i = 0; i < got.size(); ++i) std::cout << got[i] << (i + 1 < got.size() ? ", " : "");
        std::cout << "]\n";
        s.failed++;
    }
}
