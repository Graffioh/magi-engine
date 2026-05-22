#include "test_utils.h"
#include "ops.h"

void run_ops_tests(TestState& s) {
    Operations ops;
    (void)ops;
    (void)s;

    Tensor A({2, 2});
    Tensor B({2, 3});
    fill(A, {1.1f, 4.2f, 3, 5});
    fill(B, {1, 0.5f, 2.3f, 2, 6, 2});

    Tensor W({2});
    fill(W, {0.7f, 5});

    // matmul tests
    Tensor OUT_matmul({2, 3});
    ops.matmul(A, B, OUT_matmul);
    check(s, "matmul: (2 x 2) x (2 x 3)", OUT_matmul,
          std::vector<float>{9.5f, 25.75f, 10.93f, 13, 31.5f, 16.9f},
          std::vector<int>{2, 3});

    // rmsnorm tests
    Tensor OUT_rmsnorm({2, 2});
    ops.RMSNorm(A, W, OUT_rmsnorm);
    check(s, "rmsnorm: (2, 2) with W=(2,)", OUT_rmsnorm,
          std::vector<float>{0.250813f, 6.840351f, 0.509325f, 6.063389f},
          std::vector<int>{2, 2});
    // silu tests
    Tensor OUT_silu({2, 2});
    ops.SiLU(A, OUT_silu);
    check(s, "silu: (2, 2)", OUT_silu,
          std::vector<float>{0.825286f, 4.137949f, 2.857722f, 4.966536f},
          std::vector<int>{2, 2});

    // add / mul share a second operand
    Tensor A2({2, 2});
    fill(A2, {2, 1, 0.5f, -1});

    // add tests
    Tensor OUT_add({2, 2});
    ops.add(A, A2, OUT_add);
    check(s, "add: (2, 2) + (2, 2)", OUT_add,
          std::vector<float>{3.1f, 5.2f, 3.5f, 4.0f},
          std::vector<int>{2, 2});

    // mul tests
    Tensor OUT_mul({2, 2});
    ops.mul(A, A2, OUT_mul);
    check(s, "mul: (2, 2) * (2, 2)", OUT_mul,
          std::vector<float>{2.2f, 4.2f, 1.5f, -5.0f},
          std::vector<int>{2, 2});

    // matmul: identity (A @ I == A)
    Tensor A_id({3, 3});
    Tensor I3({3, 3});
    Tensor OUT_id({3, 3});
    fill(A_id, {1, 2, 3, 4, 5, 6, 7, 8, 9});
    fill(I3,   {1, 0, 0, 0, 1, 0, 0, 0, 1});
    ops.matmul(A_id, I3, OUT_id);
    check(s, "matmul: A @ I == A", OUT_id,
          std::vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9},
          std::vector<int>{3, 3});

    // matmul: batched rank-3, (2, 2, 3) @ (2, 3, 2) -> (2, 2, 2)
    Tensor A_b({2, 2, 3});
    Tensor B_b({2, 3, 2});
    Tensor OUT_b({2, 2, 2});
    fill(A_b, {1, 0, 0,
               0, 1, 0,
               0, 0, 1,
               1, 1, 1});
    fill(B_b, {1, 2,
               3, 4,
               5, 6,
               1, 0,
               0, 1,
               0, 0});
    ops.matmul(A_b, B_b, OUT_b);
    check(s, "matmul: batched (2,2,3) @ (2,3,2)", OUT_b,
          std::vector<float>{1, 2, 3, 4, 0, 0, 1, 1},
          std::vector<int>{2, 2, 2});

    // matmul: non-square inner dim, (1, 4) @ (4, 1) -> (1, 1)
    Tensor A_row({1, 4});
    Tensor B_col({4, 1});
    Tensor OUT_dot({1, 1});
    fill(A_row, {1, 2, 3, 4});
    fill(B_col, {1, 1, 1, 1});
    ops.matmul(A_row, B_col, OUT_dot);
    check(s, "matmul: (1,4) @ (4,1)", OUT_dot,
          std::vector<float>{10},
          std::vector<int>{1, 1});

    // rmsnorm: constant input. rms = sqrt(c^2 + eps) ≈ |c|, so OUT[i] ≈ sign(c) * W[i]
    Tensor IN_const({1, 2});
    Tensor W_const({2});
    Tensor OUT_const({1, 2});
    fill(IN_const, {3, 3});
    fill(W_const, {2, 4});
    ops.RMSNorm(IN_const, W_const, OUT_const);
    check(s, "rmsnorm: constant input", OUT_const,
          std::vector<float>{2.0f, 4.0f},
          std::vector<int>{1, 2});

    // rmsnorm: W = ones isolates normalization
    Tensor IN_norm({1, 4});
    Tensor W_ones({4});
    Tensor OUT_norm({1, 4});
    fill(IN_norm, {1, 2, 3, 4});
    fill(W_ones, {1, 1, 1, 1});
    ops.RMSNorm(IN_norm, W_ones, OUT_norm);
    check(s, "rmsnorm: W=ones, (1,4)", OUT_norm,
          std::vector<float>{0.365148372f, 0.730296745f, 1.095445117f, 1.460593489f},
          std::vector<int>{1, 4});

    // rmsnorm: batched rank-3, W shared across rows
    Tensor IN_rb({2, 2, 3});
    Tensor W_rb({3});
    Tensor OUT_rb({2, 2, 3});
    fill(IN_rb, {1, 1, 1,
                 2, 2, 2,
                 -1, 2, -1,
                 0, 0, 3});
    fill(W_rb, {1, 1, 1});
    ops.RMSNorm(IN_rb, W_rb, OUT_rb);
    check(s, "rmsnorm: batched (2,2,3)", OUT_rb,
          std::vector<float>{1, 1, 1,
                             1, 1, 1,
                             -0.707106781f, 1.414213562f, -0.707106781f,
                             0, 0, 1.732050808f},
          std::vector<int>{2, 2, 3});

    // silu boundary + saturation
    Tensor IN_silu({1, 5});
    Tensor OUT_silu_b({1, 5});
    fill(IN_silu, {0.0f, 2.0f, -2.0f, 20.0f, -20.0f});
    ops.SiLU(IN_silu, OUT_silu_b);
    check(s, "silu: 0, ±2, ±20", OUT_silu_b,
          std::vector<float>{0.0f, 1.761594156f, -0.238405844f, 20.0f, 0.0f},
          std::vector<int>{1, 5});

    // softmax: 1D row, hand-computed
    Tensor IN_sm({1, 3});
    Tensor OUT_sm({1, 3});
    fill(IN_sm, {1, 2, 3});
    ops.softmax(IN_sm, OUT_sm);
    check(s, "softmax: (1,3) hand-computed", OUT_sm,
          std::vector<float>{0.090030573f, 0.244728471f, 0.665240956f},
          std::vector<int>{1, 3});

    // softmax: numerical stability. shift-by-max means [1000,1001,1002] == [0,1,2]
    Tensor IN_sm_big({1, 3});
    Tensor OUT_sm_big({1, 3});
    fill(IN_sm_big, {1000, 1001, 1002});
    ops.softmax(IN_sm_big, OUT_sm_big);
    check(s, "softmax: numerical stability (large inputs)", OUT_sm_big,
          std::vector<float>{0.090030573f, 0.244728471f, 0.665240956f},
          std::vector<int>{1, 3});

    // softmax: sum-to-1 invariant on a wider row (no hand-computed values needed)
    Tensor IN_sm_sum({1, 5});
    Tensor OUT_sm_sum({1, 5});
    fill(IN_sm_sum, {-1, 0, 1, 2, 3});
    ops.softmax(IN_sm_sum, OUT_sm_sum);
    {
        const float* p = OUT_sm_sum.data_ptr();
        float total = 0;
        for (int i = 0; i < 5; ++i) total += p[i];
        Tensor SUM({1});
        SUM.data_ptr()[0] = total;
        check(s, "softmax: row sums to 1", SUM,
              std::vector<float>{1.0f},
              std::vector<int>{1});
    }

    // softmax: batched, each row independent
    Tensor IN_sm_b({2, 3});
    Tensor OUT_sm_b({2, 3});
    fill(IN_sm_b, {1, 2, 3,
                   3, 1, 2});
    ops.softmax(IN_sm_b, OUT_sm_b);
    check(s, "softmax: batched (2,3)", OUT_sm_b,
          std::vector<float>{0.090030573f, 0.244728471f, 0.665240956f,
                             0.665240956f, 0.090030573f, 0.244728471f},
          std::vector<int>{2, 3});

    // softmax: argmax preserved
    Tensor IN_sm_am({1, 4});
    Tensor OUT_sm_am({1, 4});
    fill(IN_sm_am, {0.5f, -1.0f, 2.7f, 1.0f});
    ops.softmax(IN_sm_am, OUT_sm_am);
    {
        const float* p = OUT_sm_am.data_ptr();
        int argmax = 0;
        for (int i = 1; i < 4; ++i) if (p[i] > p[argmax]) argmax = i;
        Tensor AM({1});
        AM.data_ptr()[0] = (float)argmax;
        check(s, "softmax: argmax preserved (expect index 2)", AM,
              std::vector<float>{2.0f},
              std::vector<int>{1});
    }

    // add/mul: rank-3 exercises total_dim across non-2D inputs
    Tensor A3({2, 2, 2});
    Tensor B3({2, 2, 2});
    Tensor OUT_add3({2, 2, 2});
    Tensor OUT_mul3({2, 2, 2});
    fill(A3, {1, 2, 3, 4, 5, 6, 7, 8});
    fill(B3, {8, 7, 6, 5, 4, 3, 2, 1});
    ops.add(A3, B3, OUT_add3);
    ops.mul(A3, B3, OUT_mul3);
    check(s, "add: rank-3 (2,2,2)", OUT_add3,
          std::vector<float>{9, 9, 9, 9, 9, 9, 9, 9},
          std::vector<int>{2, 2, 2});
    check(s, "mul: rank-3 (2,2,2)", OUT_mul3,
          std::vector<float>{8, 14, 18, 20, 20, 18, 14, 8},
          std::vector<int>{2, 2, 2});
}
