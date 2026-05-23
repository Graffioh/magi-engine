#pragma once

#include <vector>

#include "tensor.h"

class LinearLayer {
private:
  Tensor W; // (out_features, in_features)
public:
  LinearLayer(Tensor W);
  void forward(const Tensor &IN, Tensor &OUT) const;
};
