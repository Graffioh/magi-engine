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
}
