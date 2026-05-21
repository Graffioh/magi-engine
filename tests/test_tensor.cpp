#include "test_utils.h"
#include "tensor.h"

void run_tensor_tests(TestState& s) {
    (void)s;

    Tensor A({2, 2});
    Tensor B({2, 3});

    check(s, "strides computation for A", A.get_strides(), std::vector<int>{2,1});
    check(s, "strides computation for B", B.get_strides(), std::vector<int>{3,1});
}
