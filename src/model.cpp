#include "model.h"

#include <utility>  // std::swap

Model::Model(EmbeddingLayer                embed,
             std::vector<TransformerBlock> blocks,
             RMSNormLayer                  final_norm,
             LinearLayer                   lm_head,
             RopeCache                     rc,
             ModelConfig                   config) :
    embed_(std::move(embed)),
    blocks_(std::move(blocks)),
    final_norm_(std::move(final_norm)),
    lm_head_(std::move(lm_head)),
    rc_(std::move(rc)),
    config_(std::move(config)) {}

void Model::forward(const std::vector<int>& ids, Tensor& logits) const {
    // ping pong buffers before arena allocators
    Tensor x({ static_cast<int>(ids.size()), config_.hidden_size });
    Tensor tmp({ static_cast<int>(ids.size()), config_.hidden_size });

    embed_.forward(ids, x);

    for (const auto& block : blocks_) {
        block.forward(x, tmp, 0, rc_);
        std::swap(x, tmp);
    }

    final_norm_.forward(x, tmp);
    lm_head_.forward(tmp, logits);
}
