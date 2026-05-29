#include "layers.h"

#include "ops.h"

#include <assert.h>

LinearLayer::LinearLayer(Tensor W) : W_(std::move(W)) {}

void LinearLayer::forward(const Tensor& IN, Tensor& OUT) const {
    ops::matmul(IN, W_, OUT);
}

EmbeddingLayer::EmbeddingLayer(Tensor We) : We_(std::move(We)) {}

void EmbeddingLayer::forward(const std::vector<int>& token_ids, Tensor& OUT) const {
    ops::embed(We_, token_ids, OUT);
}

MLP::MLP(LinearLayer gate, LinearLayer up, LinearLayer down) :
    gate_(std::move(gate)),
    up_(std::move(up)),
    down_(std::move(down)) {}

void MLP::forward(const Tensor& IN, Tensor& OUT) const {
    assert(IN.rank() == 2);  // CPU inference engine => batch = 1 during computation (since it's sequential)
    std::vector<int> tmp_shape = { IN.dim(0), gate_.get_weight_out_features() };

    // GATE = IN @ W_g^T
    Tensor GATE(tmp_shape);
    gate_.forward(IN, GATE);

    // GATE = IN @ W_u^T
    Tensor UP(tmp_shape);
    up_.forward(IN, UP);

    ops::SiLU(GATE, GATE);

    // GATE = SiLU(GATE) ° UP
    ops::mul(GATE, UP, GATE);

    // OUT = GATE @ W_d^T
    down_.forward(GATE, OUT);
}
