#include "layers.h"
#include "ops.h"

LinearLayer::LinearLayer(Tensor W) : W_(std::move(W)) {}

void LinearLayer::forward(const Tensor &IN, Tensor &OUT) const { ops::matmul(IN, W_, OUT); }

MLP::MLP(LinearLayer gate, LinearLayer up, LinearLayer down)
    : gate_(std::move(gate)), up_(std::move(up)), down_(std::move(down)) {}

void MLP::forward(const Tensor &IN, Tensor &OUT) const {}
