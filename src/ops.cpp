#include "ops.h"

#include <algorithm>
#include <cassert>
#include <cmath>

namespace {

bool can_matmul(const Tensor& A, const Tensor& B) {
    if (A.rank() != B.rank()) {
        return false;
    }

    if (A.rank() < 2 || B.rank() < 2) {
        return false;
    }

    for (int i = 0; i < A.rank() - 2; ++i) {
        if (A.dim(i) != B.dim(i)) {
            return false;
        }
    }

    // B is transposed during in matmul
    if (A.dim(-1) != B.dim(-1)) {
        return false;
    }

    return true;
}

// matmul kernel
void matmul_2d(const float* A_data, const float* B_data, float* OUT_data, int M, int K, int N) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0;
            for (int k = 0; k < K; ++k) {
                sum += A_data[i * K + k] * B_data[j * K + k];
            }
            OUT_data[i * N + j] = sum;
        }
    }
}

// rmsnorm kernel
void RMSNorm_1d(const float* IN_data, const float* W_data, float* OUT_data, int dim, float eps) {
    // mean square aggregation
    float sum = 0;
    for (int i = 0; i < dim; ++i) {
        sum += IN_data[i] * IN_data[i];
    }

    // scalar mul to out
    float rms     = sqrtf(sum / dim + eps);
    float inv_rms = 1.0f / rms;
    for (int i = 0; i < dim; ++i) {
        OUT_data[i] = IN_data[i] * inv_rms * W_data[i];
    }
}

void compute_rope_pair(float* tensor_data, int head_base, int pair, const float cos_theta, const float sin_theta) {
    int   a_pos        = head_base + 2 * pair;
    int   b_pos        = head_base + (2 * pair) + 1;
    float t_a_og       = tensor_data[a_pos];
    float t_b_og       = tensor_data[b_pos];
    tensor_data[a_pos] = t_a_og * cos_theta - t_b_og * sin_theta;
    tensor_data[b_pos] = t_a_og * sin_theta + t_b_og * cos_theta;
}

void apply_rope(float*           tensor_data,
                const int        seq_len,
                const int        n_heads,
                const int        head_dim,
                const int        start_pos,
                const RopeCache& rc) {
    for (int s = 0; s < seq_len; ++s) {
        const int pos = start_pos + s;
        for (int h = 0; h < n_heads; ++h) {
            const int head_base = ((s * n_heads) + h) * head_dim;
            for (int p = 0; p < head_dim / 2; ++p) {
                const float cos_theta = rc.cos_cache()[pos * (head_dim / 2) + p];
                const float sin_theta = rc.sin_cache()[pos * (head_dim / 2) + p];

                // t_a' = t_a*cos(theta) - t_b*sin(theta)
                // t_b' = t_a*sin(theta) + t_b*cos(theta)
                compute_rope_pair(tensor_data, head_base, p, cos_theta, sin_theta);
            }
        }
    }
}

}  // namespace

namespace ops {

// OUT[i, j] = sum_k A[i, k] * B[j, k]
// Row-major computation. B needs to be un-transposed (natural layout).
void matmul(const Tensor& A, const Tensor& B, Tensor& OUT) {
    assert(can_matmul(A, B));

    // A is (..., M, K), B is (..., N, K)
    const int M = A.dim(-2);
    const int K = A.dim(-1);  // === B.dim(-1)
    const int N = B.dim(-2);

    // number of independent 2D matmuls = product of all leading dims (everything before the last two)
    int num_matrices = 1;
    for (int i = 0; i + 2 < A.rank(); ++i) {
        num_matrices *= A.dim(i);
    }

    const float* A_data   = A.data_ptr();
    const float* B_data   = B.data_ptr();
    float*       OUT_data = OUT.data_ptr();

    // run one 2D matmul per leading-dim slice
    for (int m = 0; m < num_matrices; ++m) {
        matmul_2d(A_data + m * (M * K), B_data + m * (N * K), OUT_data + m * (M * N), M, K, N);
    }
}

// rms  = sqrt((1/n) * sum_j IN[j]^2 + eps)
// OUT[i] = (IN[i] / rms) * W[i]
void RMSNorm(const Tensor& IN, const Tensor& W, Tensor& OUT, float eps) {
    int hidden_dim = IN.dim(-1);
    int num_rows   = 1;
    for (int i = 0; i + 1 < IN.rank(); ++i) {
        num_rows *= IN.dim(i);
    }

    const float* in_data  = IN.data_ptr();
    const float* w_data   = W.data_ptr();
    float*       out_data = OUT.data_ptr();

    // weight is shared across rows (size hidden_dim), input/output advance per row
    for (int r = 0; r < num_rows; ++r) {
        RMSNorm_1d(in_data + r * hidden_dim, w_data, out_data + r * hidden_dim, hidden_dim, eps);
    }
}

// SiLU(x) = x * sigmoid(x) = x / (1 + e^(-x))
void SiLU(const Tensor& IN, Tensor& OUT) {
    assert(IN.shape() == OUT.shape());

    const float* in_data  = IN.data_ptr();
    float*       out_data = OUT.data_ptr();

    int n = IN.num_elements();

    for (int i = 0; i < n; ++i) {
        out_data[i] = in_data[i] / (1.0f + expf(-in_data[i]));
    }
}

// OUT[i] = A[i] + B[i]   (elementwise)
void add(const Tensor& A, const Tensor& B, Tensor& OUT) {
    assert(A.shape() == B.shape());
    assert(A.shape() == OUT.shape());

    int n = A.num_elements();

    const float* a_data   = A.data_ptr();
    const float* b_data   = B.data_ptr();
    float*       out_data = OUT.data_ptr();
    for (int i = 0; i < n; ++i) {
        out_data[i] = a_data[i] + b_data[i];
    }
}

// OUT[i] = A[i] * B[i]   (elementwise / Hadamard product)
void mul(const Tensor& A, const Tensor& B, Tensor& OUT) {
    assert(A.shape() == B.shape());
    assert(A.shape() == OUT.shape());

    int n = A.num_elements();

    const float* a_data   = A.data_ptr();
    const float* b_data   = B.data_ptr();
    float*       out_data = OUT.data_ptr();
    for (int i = 0; i < n; ++i) {
        out_data[i] = a_data[i] * b_data[i];
    }
}

// m = max_j IN[j]
// softmax(IN)_i = e^(IN[i] - m) / sum_j e^(IN[j] - m)   (along last dim)
void softmax(const Tensor& IN, Tensor& OUT) {
    int num_rows = 1;
    for (int i = 0; i + 1 < IN.rank(); ++i) {
        num_rows *= IN.dim(i);
    }

    const float* in_data  = IN.data_ptr();
    float*       out_data = OUT.data_ptr();
    int          last_dim = IN.dim(-1);
    for (int r = 0; r < num_rows; ++r) {
        const float* in_row_dim = in_data + r * last_dim;
        float        max_val    = *std::max_element(in_row_dim, in_row_dim + last_dim);

        float sum_j = 0;
        for (int j = 0; j < last_dim; ++j) {
            float e                    = expf(in_data[r * last_dim + j] - max_val);
            out_data[r * last_dim + j] = e;
            sum_j += e;
        }

        for (int i = 0; i < last_dim; ++i) {
            out_data[r * last_dim + i] /= sum_j;
        }
    }
}

// OUT[i] = W[id]
void embed(const std::vector<int>& ids, const Tensor& We, Tensor& OUT) {
    const int hidden_dim = We.dim(1);

    const float* w_data   = We.data_ptr();
    float*       out_data = OUT.data_ptr();
    for (size_t i = 0; i < ids.size(); ++i) {
        const int id = ids[i];
        assert(id >= 0 && id < We.dim(0));

        std::memcpy(&out_data[i * hidden_dim], &w_data[id * hidden_dim], hidden_dim * sizeof(float));
    }
}

// for each token, we pick an head
//   for each head, we pick a pair of head dimensions (in Q and K)
//     for each pair of head dimensions, we perform magic with rotations
//
// Note: with GQA, we need to take in count that multiple Qs can share the same KV -> (#Q_heads > #KV_heads)
void RoPE(Tensor& Q, Tensor& K, int head_dim, int start_pos, const RopeCache& rc) {
    assert(head_dim % 2 == 0);

    int seq_len    = Q.dim(0);
    int n_heads    = Q.dim(-1) / head_dim;
    int n_kv_heads = K.dim(-1) / head_dim;

    apply_rope(Q.data_ptr(), seq_len, n_heads, head_dim, start_pos, rc);
    apply_rope(K.data_ptr(), seq_len, n_kv_heads, head_dim, start_pos, rc);
}

}  // namespace ops
