#pragma once

#include <vector>

class Tensor {
    std::vector<float> data;
    std::vector<int> shape;
    std::vector<int> strides;

public:
    Tensor(std::vector<int> shape);

    const float* data_ptr() const { return data.data(); };
    float* data_ptr() { return data.data(); };

    const std::vector<int>& get_shape() const { return shape; }
    int get_rank() const { return shape.size(); };
    int dim(int axis) const { return shape[axis < 0 ? shape.size() + axis : axis]; }
    int total_dim() const { int res = 0; for (int d : shape) res *= d; }
    const std::vector<int>& get_strides() const { return strides; }
};