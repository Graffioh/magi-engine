#include "tensor.h"

#include <iostream>

static void print_shape(const Tensor& t) {
    std::cout << "  shape   = [";
    for (size_t i = 0; i < t.get_shape().size(); ++i) {
        std::cout << t.get_shape()[i];
        if (i + 1 < t.get_shape().size()) std::cout << ", ";
    }
    std::cout << "]\n";

    std::cout << "  strides = [";
    for (size_t i = 0; i < t.get_strides().size(); ++i) {
        std::cout << t.get_strides()[i];
        if (i + 1 < t.get_strides().size()) std::cout << ", ";
    }
    std::cout << "]\n";
}

int main() {
    std::cout << "=== 1D tensor, shape [5] ===\n";
    Tensor v({5});
    print_shape(v);
    for (int i = 0; i < 5; ++i) v.write(i, static_cast<float>(i * 10));
    std::cout << "  values  = [";
    for (int i = 0; i < 5; ++i) {
        std::cout << v(i);
        if (i + 1 < 5) std::cout << ", ";
    }
    std::cout << "]\n\n";

    std::cout << "=== 2D tensor, shape [2, 3] ===\n";
    Tensor m({2, 3});
    print_shape(m);
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 3; ++j)
            m.write(i, j, static_cast<float>(i * 10 + j));
    std::cout << "  values:\n";
    for (int i = 0; i < 2; ++i) {
        std::cout << "    [";
        for (int j = 0; j < 3; ++j) {
            std::cout << m(i, j);
            if (j + 1 < 3) std::cout << ", ";
        }
        std::cout << "]\n";
    }
    std::cout << "\n";

    std::cout << "=== 3D tensor, shape [2, 3, 4] ===\n";
    Tensor t({2, 3, 4});
    print_shape(t);
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 3; ++j)
            for (int k = 0; k < 4; ++k)
                t.write(i, j, k, static_cast<float>(i * 100 + j * 10 + k));

    std::cout << "  spot checks:\n";
    std::cout << "    t(0, 0, 0) = " << t(0, 0, 0) << "  (expected 0)\n";
    std::cout << "    t(0, 0, 3) = " << t(0, 0, 3) << "  (expected 3)\n";
    std::cout << "    t(0, 2, 0) = " << t(0, 2, 0) << "  (expected 20)\n";
    std::cout << "    t(1, 0, 0) = " << t(1, 0, 0) << "  (expected 100)\n";
    std::cout << "    t(1, 2, 3) = " << t(1, 2, 3) << "  (expected 123)\n";

    return 0;
}
