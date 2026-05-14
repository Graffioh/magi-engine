#pragma once

#include <vector>

struct Tensor {
    std::vector<float> data;
    int rows;
    int cols;

    float& operator()(int r, int c) {
        return data[r * cols + c];
    }

    const float& operator()(int r, int c) const {
        return data[r * cols + c];
    }
};