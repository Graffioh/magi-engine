#pragma once

#include "layers.h"
#include "model_config.h"
#include "utils.h"

class Model {
  private:
    EmbeddingLayer                embed_;
    std::vector<TransformerBlock> blocks_;
    RMSNormLayer                  final_norm_;
    LinearLayer                   lm_head_;
    RopeCache                     rc_;
    ModelConfig                   config_;
  public:
    Model(EmbeddingLayer                embed,
          std::vector<TransformerBlock> blocks,
          RMSNormLayer                  final_norm,
          LinearLayer                   lm_head,
          RopeCache                     rc,
          ModelConfig                   config);
    void             forward(const std::vector<int>& ids, Tensor& logits) const;
    std::vector<int> generate(std::vector<int> ids, int max_new_tokens) const;
};
