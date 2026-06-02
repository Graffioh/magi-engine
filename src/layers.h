#pragma once

#include "model_config.h"
#include "tensor.h"
#include "utils.h"

class LinearLayer {
  private:
    Tensor W_;  // (out_features, in_features)
  public:
    LinearLayer(Tensor W);
    void forward(const Tensor& IN, Tensor& OUT) const;

    // this is W^T, so we need to get the first dimension for out_features
    int get_weight_out_features() const { return W_.dim(0); };
};

class EmbeddingLayer {
  private:
    Tensor We_;
  public:
    EmbeddingLayer(Tensor We);
    void forward(const std::vector<int>& token_ids, Tensor& OUT) const;
};

class RMSNormLayer {
  private:
    Tensor W_;     // (hidden_size,) per-channel scale
    float  eps_;
  public:
    RMSNormLayer(Tensor W, float eps);
    void forward(const Tensor& IN, Tensor& OUT) const;
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

class AttentionLayer {
  private:
    LinearLayer q_proj_, k_proj_, v_proj_, out_proj_;
    int         head_dim_, n_heads_, n_kv_heads_;

  public:
    AttentionLayer(const ModelConfig& config,
                   LinearLayer        q_proj,
                   LinearLayer        k_proj,
                   LinearLayer        v_proj,
                   LinearLayer        out_proj);
    void forward(const Tensor& IN, Tensor& OUT, const int start_pos, const RopeCache& rc) const;
};

class TransformerBlock {
  private:
    AttentionLayer attn_;
    MLP            mlp_;
    RMSNormLayer   attn_norm_;  // applied before attention
    RMSNormLayer   ffn_norm_;   // applied before the MLP

  public:
    TransformerBlock(AttentionLayer attn, MLP mlp, RMSNormLayer attn_norm, RMSNormLayer ffn_norm);
    void forward(const Tensor& IN, Tensor& OUT, const int start_pos, const RopeCache& rc) const;
};
