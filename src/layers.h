#pragma once

#include "tensor.h"

#include <vector>

struct LinearLayer {
  Tensor W;
  std::vector<float> bias;

  LinearLayer(int input_dim, int output_dim);
  void forward(const Tensor& IN, Tensor& OUT) const;
};

struct GatedMLP {
  LinearLayer gate_proj;
  LinearLayer up_proj;
  LinearLayer down_proj;

  GatedMLP(int hidden_size, int intermediate_size);
  void forward(const Tensor& IN, Tensor& OUT) const;
};
