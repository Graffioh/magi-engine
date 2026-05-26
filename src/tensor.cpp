#include "tensor.h"

#include <cassert>
#include <functional>
#include <memory>
#include <numeric>
#include <vector>

void Tensor::compute_strides() {
    strides_.resize(shape_.size(), 1);
    for (int i = strides_.size() - 2; i >= 0; --i) {
        strides_[i] = strides_[i + 1] * shape_[i + 1];
    }
}

// tensor with heap-allocated data.
Tensor::Tensor(std::vector<int> shape) : shape_(shape) {
    // accumulate shape values by multiplication
    int total = std::accumulate(shape_.begin(), shape_.end(), 1, std::multiplies<int>());

    // created shared buffer (on heap)
    auto buf = std::make_shared<float[]>(total);
    data_    = buf.get();
    owner_   = buf;

    compute_strides();
}

Tensor::Tensor(std::vector<int> shape, const float* data, std::shared_ptr<void> owner) :
    data_(const_cast<float*>(data)),
    shape_(std::move(shape)),
    owner_(std::move(owner)) {
    compute_strides();
}
