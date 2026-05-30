#include "ops.h"
#include "test_utils.h"

void run_ops_tests(TestState& s) {
    (void) s;

    Tensor A({ 2, 2 });
    Tensor B({ 3, 2 });
    fill(A, { 1.1f, 4.2f, 3, 5 });
    // B logical is (K=2, N=3) = [[1, 0.5, 2.3], [2, 6, 2]]; passed transposed as (N=3, K=2).
    fill(B, { 1, 2, 0.5f, 6, 2.3f, 2 });

    Tensor W({ 2 });
    fill(W, { 0.7f, 5 });

    // matmul tests
    Tensor OUT_matmul({ 2, 3 });
    ops::matmul(A, B, OUT_matmul);
    check(s, "matmul: (2 x 2) x (2 x 3)", OUT_matmul, std::vector<float>{ 9.5f, 25.75f, 10.93f, 13, 31.5f, 16.9f },
          std::vector<int>{ 2, 3 });

    // rmsnorm tests
    Tensor OUT_rmsnorm({ 2, 2 });
    ops::RMSNorm(A, W, OUT_rmsnorm);
    check(s, "rmsnorm: (2, 2) with W=(2,)", OUT_rmsnorm,
          std::vector<float>{ 0.250813f, 6.840351f, 0.509325f, 6.063389f }, std::vector<int>{ 2, 2 });
    // silu tests
    Tensor OUT_silu({ 2, 2 });
    ops::SiLU(A, OUT_silu);
    check(s, "silu: (2, 2)", OUT_silu, std::vector<float>{ 0.825286f, 4.137949f, 2.857722f, 4.966536f },
          std::vector<int>{ 2, 2 });

    // add / mul share a second operand
    Tensor A2({ 2, 2 });
    fill(A2, { 2, 1, 0.5f, -1 });

    // add tests
    Tensor OUT_add({ 2, 2 });
    ops::add(A, A2, OUT_add);
    check(s, "add: (2, 2) + (2, 2)", OUT_add, std::vector<float>{ 3.1f, 5.2f, 3.5f, 4.0f }, std::vector<int>{ 2, 2 });

    // mul tests
    Tensor OUT_mul({ 2, 2 });
    ops::mul(A, A2, OUT_mul);
    check(s, "mul: (2, 2) * (2, 2)", OUT_mul, std::vector<float>{ 2.2f, 4.2f, 1.5f, -5.0f }, std::vector<int>{ 2, 2 });

    // matmul: identity (A @ I == A)
    Tensor A_id({ 3, 3 });
    Tensor I3({ 3, 3 });
    Tensor OUT_id({ 3, 3 });
    fill(A_id, { 1, 2, 3, 4, 5, 6, 7, 8, 9 });
    fill(I3, { 1, 0, 0, 0, 1, 0, 0, 0, 1 });
    ops::matmul(A_id, I3, OUT_id);
    check(s, "matmul: A @ I == A", OUT_id, std::vector<float>{ 1, 2, 3, 4, 5, 6, 7, 8, 9 }, std::vector<int>{ 3, 3 });

    // matmul: batched rank-3, A=(2,2,3) @ B=(2,2,3) (B transposed: N=2, K=3) -> (2, 2, 2)
    Tensor A_b({ 2, 2, 3 });
    Tensor B_b({ 2, 2, 3 });
    Tensor OUT_b({ 2, 2, 2 });
    fill(A_b, { 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1 });
    // B logical per batch is (K=3, N=2); passed transposed as (N=2, K=3).
    // batch0 logical [[1,2],[3,4],[5,6]] -> transposed [[1,3,5],[2,4,6]]
    // batch1 logical [[1,0],[0,1],[0,0]] -> transposed [[1,0,0],[0,1,0]]
    fill(B_b, { 1, 3, 5, 2, 4, 6, 1, 0, 0, 0, 1, 0 });
    ops::matmul(A_b, B_b, OUT_b);
    check(s, "matmul: batched (2,2,3) @ (2,2,3) [B transposed]", OUT_b, std::vector<float>{ 1, 2, 3, 4, 0, 0, 1, 1 },
          std::vector<int>{ 2, 2, 2 });

    // matmul: non-square inner dim, A=(1, 4) @ B=(1, 4) (B transposed: N=1, K=4) -> (1, 1)
    Tensor A_row({ 1, 4 });
    Tensor B_col({ 1, 4 });
    Tensor OUT_dot({ 1, 1 });
    fill(A_row, { 1, 2, 3, 4 });
    // B logical (K=4, N=1) = [[1],[1],[1],[1]]; transposed (N=1, K=4) = [[1,1,1,1]].
    fill(B_col, { 1, 1, 1, 1 });
    ops::matmul(A_row, B_col, OUT_dot);
    check(s, "matmul: (1,4) @ (1,4) [B transposed]", OUT_dot, std::vector<float>{ 10 }, std::vector<int>{ 1, 1 });

    // rmsnorm: constant input. rms = sqrt(c^2 + eps) ≈ |c|, so OUT[i] ≈ sign(c) * W[i]
    Tensor IN_const({ 1, 2 });
    Tensor W_const({ 2 });
    Tensor OUT_const({ 1, 2 });
    fill(IN_const, { 3, 3 });
    fill(W_const, { 2, 4 });
    ops::RMSNorm(IN_const, W_const, OUT_const);
    check(s, "rmsnorm: constant input", OUT_const, std::vector<float>{ 2.0f, 4.0f }, std::vector<int>{ 1, 2 });

    // rmsnorm: W = ones isolates normalization
    Tensor IN_norm({ 1, 4 });
    Tensor W_ones({ 4 });
    Tensor OUT_norm({ 1, 4 });
    fill(IN_norm, { 1, 2, 3, 4 });
    fill(W_ones, { 1, 1, 1, 1 });
    ops::RMSNorm(IN_norm, W_ones, OUT_norm);
    check(s, "rmsnorm: W=ones, (1,4)", OUT_norm,
          std::vector<float>{ 0.365148372f, 0.730296745f, 1.095445117f, 1.460593489f }, std::vector<int>{ 1, 4 });

    // rmsnorm: batched rank-3, W shared across rows
    Tensor IN_rb({ 2, 2, 3 });
    Tensor W_rb({ 3 });
    Tensor OUT_rb({ 2, 2, 3 });
    fill(IN_rb, { 1, 1, 1, 2, 2, 2, -1, 2, -1, 0, 0, 3 });
    fill(W_rb, { 1, 1, 1 });
    ops::RMSNorm(IN_rb, W_rb, OUT_rb);
    check(s, "rmsnorm: batched (2,2,3)", OUT_rb,
          std::vector<float>{ 1, 1, 1, 1, 1, 1, -0.707106781f, 1.414213562f, -0.707106781f, 0, 0, 1.732050808f },
          std::vector<int>{ 2, 2, 3 });

    // silu boundary + saturation
    Tensor IN_silu({ 1, 5 });
    Tensor OUT_silu_b({ 1, 5 });
    fill(IN_silu, { 0.0f, 2.0f, -2.0f, 20.0f, -20.0f });
    ops::SiLU(IN_silu, OUT_silu_b);
    check(s, "silu: 0, ±2, ±20", OUT_silu_b, std::vector<float>{ 0.0f, 1.761594156f, -0.238405844f, 20.0f, 0.0f },
          std::vector<int>{ 1, 5 });

    // softmax: 1D row, hand-computed
    Tensor IN_sm({ 1, 3 });
    Tensor OUT_sm({ 1, 3 });
    fill(IN_sm, { 1, 2, 3 });
    ops::softmax(IN_sm, OUT_sm);
    check(s, "softmax: (1,3) hand-computed", OUT_sm, std::vector<float>{ 0.090030573f, 0.244728471f, 0.665240956f },
          std::vector<int>{ 1, 3 });

    // softmax: numerical stability. shift-by-max means [1000,1001,1002] == [0,1,2]
    Tensor IN_sm_big({ 1, 3 });
    Tensor OUT_sm_big({ 1, 3 });
    fill(IN_sm_big, { 1000, 1001, 1002 });
    ops::softmax(IN_sm_big, OUT_sm_big);
    check(s, "softmax: numerical stability (large inputs)", OUT_sm_big,
          std::vector<float>{ 0.090030573f, 0.244728471f, 0.665240956f }, std::vector<int>{ 1, 3 });

    // softmax: sum-to-1 invariant on a wider row (no hand-computed values needed)
    Tensor IN_sm_sum({ 1, 5 });
    Tensor OUT_sm_sum({ 1, 5 });
    fill(IN_sm_sum, { -1, 0, 1, 2, 3 });
    ops::softmax(IN_sm_sum, OUT_sm_sum);
    {
        const float* p     = OUT_sm_sum.data_ptr();
        float        total = 0;
        for (int i = 0; i < 5; ++i) {
            total += p[i];
        }
        Tensor SUM({ 1 });
        SUM.data_ptr()[0] = total;
        check(s, "softmax: row sums to 1", SUM, std::vector<float>{ 1.0f }, std::vector<int>{ 1 });
    }

    // softmax: batched, each row independent
    Tensor IN_sm_b({ 2, 3 });
    Tensor OUT_sm_b({ 2, 3 });
    fill(IN_sm_b, { 1, 2, 3, 3, 1, 2 });
    ops::softmax(IN_sm_b, OUT_sm_b);
    check(s, "softmax: batched (2,3)", OUT_sm_b,
          std::vector<float>{ 0.090030573f, 0.244728471f, 0.665240956f, 0.665240956f, 0.090030573f, 0.244728471f },
          std::vector<int>{ 2, 3 });

    // softmax: argmax preserved
    Tensor IN_sm_am({ 1, 4 });
    Tensor OUT_sm_am({ 1, 4 });
    fill(IN_sm_am, { 0.5f, -1.0f, 2.7f, 1.0f });
    ops::softmax(IN_sm_am, OUT_sm_am);
    {
        const float* p      = OUT_sm_am.data_ptr();
        int          argmax = 0;
        for (int i = 1; i < 4; ++i) {
            if (p[i] > p[argmax]) {
                argmax = i;
            }
        }
        Tensor AM({ 1 });
        AM.data_ptr()[0] = (float) argmax;
        check(s, "softmax: argmax preserved (expect index 2)", AM, std::vector<float>{ 2.0f }, std::vector<int>{ 1 });
    }

    // add/mul: rank-3 exercises total_dim across non-2D inputs
    Tensor A3({ 2, 2, 2 });
    Tensor B3({ 2, 2, 2 });
    Tensor OUT_add3({ 2, 2, 2 });
    Tensor OUT_mul3({ 2, 2, 2 });
    fill(A3, { 1, 2, 3, 4, 5, 6, 7, 8 });
    fill(B3, { 8, 7, 6, 5, 4, 3, 2, 1 });
    ops::add(A3, B3, OUT_add3);
    ops::mul(A3, B3, OUT_mul3);
    check(s, "add: rank-3 (2,2,2)", OUT_add3, std::vector<float>{ 9, 9, 9, 9, 9, 9, 9, 9 },
          std::vector<int>{ 2, 2, 2 });
    check(s, "mul: rank-3 (2,2,2)", OUT_mul3, std::vector<float>{ 8, 14, 18, 20, 20, 18, 14, 8 },
          std::vector<int>{ 2, 2, 2 });

    // RoPE toy dims -- must match HEAD_DIM/N_HEADS/N_KV_HEADS in gen_golden_weights.py.
    constexpr int HEAD_DIM   = 4;
    constexpr int N_HEADS    = 2;
    constexpr int N_KV_HEADS = 1;

    // RoPE: pos=0 identity. A single token sits at position 0, where theta=0 for
    // every pair (cos=1, sin=0), so the rotation is the identity. seq_len MUST be
    // 1: rows beyond position 0 rotate by a non-zero angle and would not match.
    {
        Tensor               Q({ 1, N_HEADS * HEAD_DIM });     // (seq_len=1, 8)
        Tensor               K({ 1, N_KV_HEADS * HEAD_DIM });  // (seq_len=1, 4)
        std::vector<float>   q_vals = { 0.1f, -0.2f, 0.3f, -0.4f, 0.5f, -0.6f, 0.7f, -0.8f };
        std::vector<float>   k_vals = { 1.5f, -2.5f, 3.5f, -4.5f };
        fill(Q, q_vals);
        fill(K, k_vals);
        ops::RoPE(Q, K, HEAD_DIM, /*start_pos=*/0);
        // theta=0 -> bit-exact identity, but reuse check()'s default tolerance.
        check(s, "rope: pos=0 identity (Q)", Q, q_vals, std::vector<int>{ 1, N_HEADS * HEAD_DIM });
        check(s, "rope: pos=0 identity (K)", K, k_vals, std::vector<int>{ 1, N_KV_HEADS * HEAD_DIM });
    }

    // RoPE: norm preservation. Rotation is orthogonal, so the total sum-of-squares
    // of Q (and of K) is invariant under RoPE. A non-zero start_pos makes the
    // rotations non-trivial; this is the test that catches sign errors in the
    // pair update (a swapped sign breaks orthogonality and changes the norm).
    {
        Tensor Q({ 3, N_HEADS * HEAD_DIM });     // (T=3, 8)
        Tensor K({ 3, N_KV_HEADS * HEAD_DIM });  // (T=3, 4)
        fill_random_norm(Q, /*seed=*/123);
        fill_random_norm(K, /*seed=*/456);

        auto sum_sq = [](const Tensor& t) {
            const float* p     = t.data_ptr();
            float        total = 0;
            for (int i = 0; i < t.num_elements(); ++i) {
                total += p[i] * p[i];
            }
            return total;
        };
        float q_before = sum_sq(Q);
        float k_before = sum_sq(K);
        ops::RoPE(Q, K, HEAD_DIM, /*start_pos=*/5);

        Tensor Q_NORM({ 1 });
        Q_NORM.data_ptr()[0] = sum_sq(Q);
        check(s, "rope: norm preserved (Q, start_pos=5)", Q_NORM, std::vector<float>{ q_before },
              std::vector<int>{ 1 });
        Tensor K_NORM({ 1 });
        K_NORM.data_ptr()[0] = sum_sq(K);
        check(s, "rope: norm preserved (K, start_pos=5)", K_NORM, std::vector<float>{ k_before },
              std::vector<int>{ 1 });
    }

    // RoPE: relative-shift invariance (the conceptual capstone). For single-head
    // q,k vectors, score(m,n) = dot(rope(q,m), rope(k,n)) depends only on the
    // relative offset m-n. We rotate a single vector by giving RoPE a (1,HEAD_DIM)
    // tensor plus a throwaway second tensor (RoPE mutates both in place, so we use
    // fresh copies each call). score(2,5) and score(4,7) both have m-n = -3.
    {
        const std::vector<float> q_base = { 0.3f, -0.7f, 0.2f, 0.9f };
        const std::vector<float> k_base = { -0.4f, 0.6f, 0.8f, -0.1f };

        // Rotate `vec_vals` (one head) at `pos` and return its dot with `other`.
        auto rope_vec = [&](const std::vector<float>& vec_vals, int pos) {
            Tensor vec({ 1, HEAD_DIM });
            Tensor dummy({ 1, HEAD_DIM });  // throwaway: RoPE rotates both args
            fill(vec, vec_vals);
            fill(dummy, std::vector<float>(HEAD_DIM, 0.0f));
            ops::RoPE(vec, dummy, HEAD_DIM, /*start_pos=*/pos);
            return std::vector<float>(vec.data_ptr(), vec.data_ptr() + HEAD_DIM);
        };
        auto score = [&](int m, int n) {
            std::vector<float> rq  = rope_vec(q_base, m);
            std::vector<float> rk  = rope_vec(k_base, n);
            float              dot = 0;
            for (int i = 0; i < HEAD_DIM; ++i) {
                dot += rq[i] * rk[i];
            }
            return dot;
        };

        Tensor SCORE({ 1 });
        SCORE.data_ptr()[0] = score(2, 5);  // m-n = -3
        check(s, "rope: relative-shift invariance score(2,5)==score(4,7)", SCORE,
              std::vector<float>{ score(4, 7) }, std::vector<int>{ 1 }, 1e-4f);  // m-n = -3
    }
}
