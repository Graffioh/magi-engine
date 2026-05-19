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
    for (size_t i = strides.size() - 2; i >= 0; --i) {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
}

float Tensor::operator()(int i) const {
    assert(shape.size() == 1 && i >= 0 && i < shape[0]);
    return data[i * strides[0]];
}

float Tensor::operator()(int i, int j) const {
    assert(shape.size() == 2 && i >= 0 && j >= 0 && i < shape[0] && j < shape[1]);
    return data[i * strides[0] + j * strides[1]];
}

float Tensor::operator()(int i, int j, int k) const {
    assert(shape.size() == 3 && i >= 0 && j >= 0 && k >= 0 && i < shape[0] && j < shape[1] && k < shape[2]);
    return data[i * strides[0] + j * strides[1] + k * strides[2]];
}

void Tensor::write(int i, float value) {
    assert(shape.size() == 1 && i >= 0 && i < shape[0]);
    data[i * strides[0]] = value;
}

void Tensor::write(int i, int j, float value) {
    assert(shape.size() == 2 && i >= 0 && j >= 0 && i < shape[0] && j < shape[1]);
    data[i * strides[0] + j * strides[1]] = value;
}

void Tensor::write(int i, int j, int k, float value) {
    assert(shape.size() == 3 && i >= 0 && j >= 0 && k >= 0 && i < shape[0] && j < shape[1] && k < shape[2]);
    data[i * strides[0] + j * strides[1] + k * strides[2]] = value;
}