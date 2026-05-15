#pragma once

#include "tensor.h"

#include <vector>

struct LinearLayer {
  Tensor W;
  std::vector<float> bias;

  LinearLayer(int input_dim, int output_dim, bool use_bias = false);
  void forward(const Tensor& IN, Tensor& OUT) const;
  bool has_bias() const { return !bias.empty(); }
};

struct GatedMLP {
  LinearLayer gate_proj;
  LinearLayer up_proj;
  LinearLayer down_proj;

  GatedMLP(int hidden_size, int intermediate_size);
  void forward(const Tensor& IN, Tensor& OUT) const;
};
