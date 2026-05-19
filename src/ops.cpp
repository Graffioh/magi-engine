#include <cassert>

#include "ops.h"

static void check_matmul_shapes(const Tensor& A, const Tensor& B) {
    std::vector<int> a_shape = A.get_shape();
    std::vector<int> b_shape = B.get_shape();

    // no sense to compare 1D since matmul would become a simple dot product
    assert(a_shape.size() >= 2 && b_shape.size() >= 2);

    // compare last dim of A with second-last dim of B (the rest of remaining dimensions if any are just how many times we do the matmul, also called batch dimensions)
    assert(a_shape[a_shape.size() - 1] == b_shape[b_shape.size() - 2]);

    // all remaining dims must be equal (stricter than numpy/pytorch 'broadcastable')
    assert(a_shape.size() == b_shape.size());
    for (size_t i = 0; i < b_shape.size() - 2; ++i) {
        assert(a_shape[i] == b_shape[i]);
    }
} 

static std::vector<int> compute_matmul_output_shape(const Tensor& A, const Tensor& B) {
    std::vector<int> a_shape = A.get_shape();
    std::vector<int> b_shape = B.get_shape();
    std::vector<int> out_shape;
    out_shape.reserve(a_shape.size());
    for (size_t i = 0; i < a_shape.size() - 2; ++i) {
        out_shape.push_back(a_shape[i]);
    }
    int M = a_shape[a_shape.size() - 2];
    int N = b_shape[b_shape.size() - 1];
    out_shape.push_back(M);
    out_shape.push_back(N);

    return out_shape; // (..., M, K) x (..., K, N) = (..., M, N)
}

void Operations::matmul(const Tensor& A, const Tensor& B, Tensor& OUT) {
    // check if A and B have compatible dimensions
    check_matmul_shapes(A, B);
    // check if OUT has the right final dimension
    assert(OUT.get_shape() == compute_matmul_output_shape(A, B));

    // ...
}