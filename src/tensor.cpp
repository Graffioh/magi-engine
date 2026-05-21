#include "tensor.h"
#include <functional>
#include <cassert>
#include <vector>
#include <numeric>

Tensor::Tensor(std::vector<int> shape): shape(shape) {
    // accumulate shape values by multiplication
    int total = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());

    data.resize(total);

    strides.resize(shape.size(), 1);
    for (int i = strides.size() - 2; i >= 0; --i) {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
}