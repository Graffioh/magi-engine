#include "ops.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <limits>

float NEG_INF_F = -std::numeric_limits<float>::infinity();

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

void apply_causal_mask(Tensor& INOUT) {
    const int M = INOUT.dim(0);
    const int N = INOUT.dim(1);
    for (int i = 0; i < M; ++i) {
        for (int j = i; j < N; ++j) {
            if (j > i) {
                INOUT.data_ptr()[i * M + j] = NEG_INF_F;
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

// m = max_j INOUT[j]
// softmax(INOUT)_i = e^(INOUT[i] - m) / sum_j e^(INOUT[j] - m)   (along last dim)
// In-place: each row is overwritten with its softmax.
void softmax(Tensor& INOUT) {
    int num_rows = 1;
    for (int i = 0; i + 1 < INOUT.rank(); ++i) {
        num_rows *= INOUT.dim(i);
    }

    float* data     = INOUT.data_ptr();
    int    last_dim = INOUT.dim(-1);
    for (int r = 0; r < num_rows; ++r) {
        float* row     = data + r * last_dim;
        float  max_val = *std::max_element(row, row + last_dim);

        float sum_j = 0;
        for (int j = 0; j < last_dim; ++j) {
            float e = expf(row[j] - max_val);
            row[j]  = e;
            sum_j += e;
        }

        for (int i = 0; i < last_dim; ++i) {
            row[i] /= sum_j;
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
void RoPE(Tensor& Q, Tensor& K, const int head_dim, const int start_pos, const RopeCache& rc) {
    assert(head_dim % 2 == 0);

    int seq_len    = Q.dim(0);
    int n_heads    = Q.dim(-1) / head_dim;
    int n_kv_heads = K.dim(-1) / head_dim;

    apply_rope(Q.data_ptr(), seq_len, n_heads, head_dim, start_pos, rc);
    apply_rope(K.data_ptr(), seq_len, n_kv_heads, head_dim, start_pos, rc);
}

// TODO: add start_pos for inference
void attn(const Tensor& Q,
          const Tensor& K,
          const Tensor& V,
          Tensor&       OUT,
          const int     n_heads,
          const int     n_kv_heads,
          const int     head_dim) {
    int T = Q.dim(0);

    float        head_dim_sqrt = sqrtf(head_dim);
    const float* q_data_ptr    = Q.data_ptr();
    const float* k_data_ptr    = K.data_ptr();
    const float* v_data_ptr    = V.data_ptr();

    // GQA: Q heads = n_heads, KV heads = n_kv_heads, n_heads > n_kv_heads
    //    for each Q head, we must pick the corresponding KV head
    //
    const int group_size = (n_heads / n_kv_heads);
    const int q_width    = n_heads * head_dim;
    const int kv_width   = n_kv_heads * head_dim;
    for (int h = 0; h < n_heads; ++h) {
        const int kv_head = (h / group_size);

        // QK^T: (T, head_dim) x (head_dim, T) -> (T, T)
        // we need to gather the heads into big tensor into single head tensor
        // TODO: first scratch buffer, then fast kernel with indexing.
        Tensor Q_head({ T, head_dim });
        Tensor K_head({ T, head_dim });
        for (int t = 0; t < T; ++t) {
            for (int d = 0; d < head_dim; ++d) {
                Q_head.data_ptr()[t * head_dim + d] = q_data_ptr[t * q_width + h * head_dim + d];
                K_head.data_ptr()[t * head_dim + d] = k_data_ptr[t * kv_width + kv_head * head_dim + d];
            }
        }

        // attn_score: matmul -> divide by sqrt(h_d) -> causal mask -> softmax
        Tensor ATTN_score({ T, T });

        matmul(Q_head, K_head, ATTN_score);

        for (int t = 0; t < T * T; ++t) {
            ATTN_score.data_ptr()[t] /= head_dim_sqrt;
        }

        apply_causal_mask(ATTN_score);

        softmax(ATTN_score);

        // attn (OUT) = attn_score * V: (T, T) x (T, head_dim) -> (T, head_dim)
        Tensor V_head({ head_dim, T });
        for (int t = 0; t < T; ++t) {
            for (int d = 0; d < head_dim; ++d) {
                V_head.data_ptr()[d * T + t] = v_data_ptr[t * kv_width + kv_head * head_dim + d];
            }
        }

        // we need to scatter the head of the small tensor into the big OUT tensor containing all the heads
        Tensor OUT_head({ T, head_dim });
        matmul(ATTN_score, V_head, OUT_head);
        for (int t = 0; t < T; ++t) {
            for (int d = 0; d < head_dim; ++d) {
                OUT.data_ptr()[t * q_width + h * head_dim + d] = OUT_head.data_ptr()[t * head_dim + d];
            }
        }
    }
}

int argmax(const Tensor& logits) {
    const int    vocab = logits.dim(-1);
    const float* row   = logits.data_ptr() + (logits.dim(0) - 1) * logits.dim(-1);

    int max_arg = 0;
    for (int j = 1; j < vocab; ++j) {
        if (row[j] > row[max_arg]) {
            max_arg = j;
        }
    }

    return max_arg;
}

}  // namespace ops
