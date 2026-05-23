#include "layers.h"
#include "ops.h"

LinearLayer::LinearLayer(Tensor W) : W(W) {}

void LinearLayer::forward(const Tensor &IN, Tensor &OUT) const { ops::matmul(IN, W, OUT); }
