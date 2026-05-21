#include <cassert>

#include "ops.h"

static void check_matmul_shapes(const Tensor& A, const Tensor& B) {
    // no sense to compare 1D since matmul would become a simple dot product
    assert(A.get_rank() >= 2 && B.get_rank() >= 2);

    // A is (..., M, K), B is (..., K, N) — the inner dim must match
    assert(A.dim(-1) == B.dim(-2));

    // all remaining dims must be equal (stricter than numpy/pytorch 'broadcastable')
    assert(A.get_rank() == B.get_rank());
    for (int i = 0; i < A.get_rank() - 2; ++i) {
        assert(A.dim(i) == B.dim(i));
    }
}

static std::vector<int> compute_matmul_output_shape(const Tensor& A, const Tensor& B) {
    // (..., M, K) x (..., K, N) = (..., M, N) — copy batch dims, then append M and N
    std::vector<int> out_shape;
    out_shape.reserve(A.get_rank());
    for (int i = 0; i < A.get_rank() - 2; ++i) {
        out_shape.push_back(A.dim(i));
    }
    out_shape.push_back(A.dim(-2));   // M
    out_shape.push_back(B.dim(-1));   // N
    return out_shape;
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

    // A is (..., M, K), B is (..., K, N)
    const int M = A.dim(-2);
    const int K = A.dim(-1);
    const int N = B.dim(-1);

    // total batch size = product of all leading dims (everything before the last two)
    int batch_size = 1;
    for (int i = 0; i + 2 < A.get_rank(); ++i) {
        batch_size *= A.dim(i);
    }

    const float* A_data = A.data_ptr();
    const float* B_data = B.data_ptr();
    float* OUT_data = OUT.data_ptr();

    // do matmul across batches
    for (size_t b = 0; b < batch_size; ++b) {
        matmul_2d(A_data + b * (M * K), B_data + b * (K * N), OUT_data + b * (M * N), M, K, N);
    }
}

void Operations::RMSNorm_1d(const float* IN_data, const float* W_data, float* OUT_data, int dim, float eps) {
    // mean square aggregation
    float sum = 0;
    for (int i = 0; i < dim; ++i) {
        sum += IN_data[i] * IN_data[i];
    }

    // scalar mul to out
    float rms = sqrtf(sum / dim + eps);
    float inv_rms = 1.0f / rms;
    for (int i = 0; i < dim; ++i) {
        OUT_data[i] = IN_data[i] * inv_rms * W_data[i];
    }
}

void Operations::RMSNorm(const Tensor& IN, const Tensor& W, Tensor& OUT, float eps) {
    int hidden_dim = IN.dim(-1);
    int batch_size = 1;
    for (int i = 0; i + 1 < IN.get_rank(); ++i) {
        batch_size *= IN.dim(i);
    }

    const float* in_data = IN.data_ptr();
    const float* w_data = W.data_ptr();
    float* out_data = OUT.data_ptr();

    // weight is shared across rows (size hidden_dim), input/output advance per row
    for (size_t b = 0; b < batch_size; ++b) {
        RMSNorm_1d(in_data + b * hidden_dim, w_data, out_data + b * hidden_dim, hidden_dim, eps);
    }
}

void Operations::SiLU(const Tensor& IN, Tensor& OUT) {
    assert(IN.get_shape() == OUT.get_shape());

    const float* in_data = IN.data_ptr();
    float* out_data = OUT.data_ptr();

    int total_dim = IN.total_dim();

    for (int i = 0; i < total_dim; ++i) {
        out_data[i] = in_data[i] / (1.0f + expf(-in_data[i]));
    }
}

void Operations::add(const Tensor& A, const Tensor& B, Tensor& OUT) {
    assert(A.get_shape() == B.get_shape());
    assert(A.get_shape() == OUT.get_shape());

    int total_dim = A.total_dim();

    const float* a_data = A.data_ptr();
    const float* b_data = B.data_ptr();
    float* out_data = OUT.data_ptr();
    for (int i = 0; i < total_dim; ++i) {
        out_data[i] = a_data[i] + b_data[i];
    }
}

void Operations::mul(const Tensor& A, const Tensor& B, Tensor& OUT) {
    assert(A.get_shape() == B.get_shape());
    assert(A.get_shape() == OUT.get_shape());

    int total_dim = A.total_dim();

    const float* a_data = A.data_ptr();
    const float* b_data = B.data_ptr();
    float* out_data = OUT.data_ptr();
    for (int i = 0; i < total_dim; ++i) {
        out_data[i] = a_data[i] * b_data[i];
    }
}