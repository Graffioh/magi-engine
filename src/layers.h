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

struct Embedding {
  Tensor W; // (vocab_size, embedding_dim)

  Embedding(int vocab_size, int embedding_size);
  void forward(const std::vector<int>& token_ids, Tensor& OUT) const;
};

struct SelfAttention {
  LinearLayer q_proj; // (hidden_size, hidden_size)
  LinearLayer k_proj; // ""
  LinearLayer v_proj; // ""
  LinearLayer o_proj; // ""

  int hidden_size;
  int num_heads;
  int num_kv_heads;
  int head_dim;

  SelfAttention(int hidden_size, int num_heads, int num_kv_heads);
  void causal_mask(Tensor& attn_scores) const;
  void apply_rope(Tensor& x, int start_pos, float theta = 10000.0f) const;
  void attn(const Tensor& q, const Tensor& k, const Tensor& v, Tensor& OUT) const;
  void forward(const Tensor& IN, Tensor& OUT) const;
};