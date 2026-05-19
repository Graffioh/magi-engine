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

void Operations::matmul_2d(const float* A_data, const float* B_data, float* OUT_data, int M, int K, int N) {
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            float sum = 0;
            for (size_t k = 0; k < K; ++k) {
                sum += A_data[i * K + k] * B_data[k * N + j];
            }
            OUT_data[i * N + j] = sum;
        }
    }
}

void Operations::matmul(const Tensor& A, const Tensor& B, Tensor& OUT) {
    // check if A and B have compatible dimensions
    check_matmul_shapes(A, B);
    // check if OUT has the right final dimension
    assert(OUT.get_shape() == compute_matmul_output_shape(A, B));

    // check overall batch size excluding matmul dimensions
    std::vector<int> a_shape = A.get_shape();
    std::vector<int> b_shape = B.get_shape();
    int M = a_shape[a_shape.size() - 2];
    int K = a_shape[a_shape.size() - 1];
    int N = b_shape[b_shape.size() - 1];
    int batch_size = 1;
    for (size_t i = 0; i + 2 < a_shape.size(); ++i) {
        batch_size *= a_shape[i];
    }

    // get second-last and last dimensions for A, B to correctly do matmul_2d
    const float* A_data = A.data_ptr();
    const float* B_data = B.data_ptr();
    float* OUT_data = OUT.data_ptr();

    // do matmul across batches
    for (size_t b = 0; b < batch_size; ++b) {
        matmul_2d(A_data + b * (M * K), B_data + b * (K * N), OUT_data + b * (M * N), M, K, N);
    }
}