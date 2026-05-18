#pragma once

#include <vector>

class Tensor {
    std::vector<float> data;
    std::vector<int> shape;
    std::vector<int> strides;

public:
    Tensor(std::vector<int> shape);

    float operator()(int i) const;
    float operator()(int i, int j) const;
    float operator()(int i, int j, int k) const;
    void write(int i, float value);
    void write(int i, int j, float value);
    void write(int i, int j, int k, float value);

    const std::vector<int>& get_shape() const { return shape; }
    const std::vector<int>& get_strides() const { return strides; }
};