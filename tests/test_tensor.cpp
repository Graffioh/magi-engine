#include "test_utils.h"
#include "tensor.h"

void run_tensor_tests(TestState& s) {
    (void)s;

    Tensor A({2, 2});
    Tensor B({2, 3});

    check(s, "strides computation for A", A.strides(), std::vector<int>{2,1});
    check(s, "strides computation for B", B.strides(), std::vector<int>{3,1});

    Tensor C({2, 3, 4});
    check(s, "strides: rank-3 (2,3,4)", C.strides(), std::vector<int>{12, 4, 1});

    Tensor D({2, 3, 4, 5});
    check(s, "strides: rank-4 (2,3,4,5)", D.strides(), std::vector<int>{60, 20, 5, 1});

    check(s, "dim(-1) of (2,3,4)", std::vector<int>{C.dim(-1)}, std::vector<int>{4});
    check(s, "dim(-2) of (2,3,4)", std::vector<int>{C.dim(-2)}, std::vector<int>{3});
    check(s, "dim(0)  of (2,3,4)", std::vector<int>{C.dim(0)}, std::vector<int>{2});

    check(s, "num_elements of (2,3,4)",   std::vector<int>{C.num_elements()},   std::vector<int>{24});
    check(s, "num_elements of (2,3,4,5)", std::vector<int>{D.num_elements()}, std::vector<int>{120});
}
