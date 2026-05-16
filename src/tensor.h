#pragma once

#include <vector>

struct Tensor {
    std::vector<float> data;
    int rows;
    int cols;
    
    Tensor(int rows = 0, int cols = 0) : rows(rows), cols(cols) {
        data.resize(rows * cols, 0.0f);
    }

    float& operator()(int r, int c) {
        return data[r * cols + c];
    }

    const float& operator()(int r, int c) const {
        return data[r * cols + c];
    }
};