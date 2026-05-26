#pragma once

#include <vector>
#include <memory>

class Tensor {
private:
    float *data_;
    std::vector<int> shape_, strides_;
    std::shared_ptr<void> owner_;

    // strides tell how many elements to skip in the flat buffers, to advance one step along each axis
    void compute_strides();

public:
    // we explicitly only allow moving, not copying (copying model weights can become a disaster x))
    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;
    Tensor(Tensor&&) = default;
    Tensor& operator=(Tensor&&) = default;

    Tensor(std::vector<int> shape);

    const float* data_ptr() const { return data_; };
    float* data_ptr() { return data_; };

    const std::vector<int>& shape() const { return shape_; }
    int rank() const { return shape_.size(); };
    int dim(int axis) const { return shape_[axis < 0 ? shape_.size() + axis : axis]; }
    int num_elements() const { int res = 1; for (int d : shape_) res *= d; return res; }
    const std::vector<int>& strides() const { return strides_; }
};
