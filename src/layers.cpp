#include "layers.h"
#include "ops.h"

LinearLayer::LinearLayer(Tensor W) : W(std::move(W)) {}

void LinearLayer::forward(const Tensor &IN, Tensor &OUT) const { ops::matmul(IN, W, OUT); }

MLP::MLP(LinearLayer gate, LinearLayer up, LinearLayer down)
    : gate(std::move(gate)), up(std::move(up)), down(std::move(down)) {}

void MLP::forward(const Tensor &IN, Tensor &OUT) const {}
