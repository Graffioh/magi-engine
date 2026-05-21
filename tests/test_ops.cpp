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

    Tensor OUT({2,3});
    ops.matmul(A, B, OUT);
    
    // matmul tests
    check(s, "matmul: (2 x 2) x (2 x 3)", OUT,
          std::vector<float>{9.5f, 25.75f, 10.93f, 13, 31.5f, 16.9f},
          std::vector<int>{2, 3});
    // TODO: rmsnorm tests
    // TODO: silu tests
    // TODO: add tests
    // TODO: mul tests
}
