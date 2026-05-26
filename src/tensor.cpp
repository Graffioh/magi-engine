#include "tensor.h"
#include <functional>
#include <cassert>
#include <memory>
#include <vector>
#include <numeric>

void Tensor::compute_strides() {
    strides_.resize(shape_.size(), 1);
    for (int i = strides_.size() - 2; i >= 0; --i) {
        strides_[i] = strides_[i + 1] * shape_[i + 1];
    }
}

// tensor with heap-allocated data.
Tensor::Tensor(std::vector<int> shape): shape_(shape) {
    // accumulate shape values by multiplication
    int total = std::accumulate(shape_.begin(), shape_.end(), 1, std::multiplies<int>());

    // created shared buffer (on heap)
    auto buf = std::make_shared<float[]>(total);
    data_ = buf.get();
    owner_ = buf;

    compute_strides();
}