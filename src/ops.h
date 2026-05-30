#pragma once

#include "tensor.h"

namespace ops {
void matmul(const Tensor& A, const Tensor& B, Tensor& OUT);
void RMSNorm(const Tensor& IN, const Tensor& W, Tensor& OUT, float eps = 1e-5f);
void SiLU(const Tensor& IN, Tensor& OUT);
void add(const Tensor& A, const Tensor& B, Tensor& OUT);
void mul(const Tensor& A, const Tensor& B, Tensor& OUT);
void softmax(const Tensor& IN, Tensor& OUT);
void embed(const std::vector<int>& ids, const Tensor& We, Tensor& OUT);
void RoPE(Tensor& Q, Tensor& K, int head_dim, int start_pos = 0, int base = 10000);
}  // namespace ops
