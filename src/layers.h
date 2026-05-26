#pragma once

#include "tensor.h"

class LinearLayer {
  private:
    Tensor W_;  // (out_features, in_features)
  public:
    LinearLayer(Tensor W);
    void forward(const Tensor& IN, Tensor& OUT) const;

    // this is W^T, so we need to get the first dimension for out_features
    int get_weight_out_features() const { return W_.dim(0); };
};

// SwiGLU MLP
class MLP {
  private:
    LinearLayer gate_;
    LinearLayer up_;
    LinearLayer down_;

  public:
    MLP(LinearLayer gate, LinearLayer up, LinearLayer down);
    void forward(const Tensor& IN, Tensor& OUT) const;
};
