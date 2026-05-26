#pragma once

#include "tensor.h"

class LinearLayer {
private:
  Tensor W_; // (out_features, in_features)
public:
  LinearLayer(Tensor W);
  void forward(const Tensor &IN, Tensor &OUT) const;
};

// SwiGLU MLP
class MLP {
private:
  LinearLayer gate_;
  LinearLayer up_;
  LinearLayer down_;

public:
  MLP(LinearLayer gate, LinearLayer up, LinearLayer down);
  void forward(const Tensor &IN, Tensor &OUT) const;
};
