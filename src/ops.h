#pragma once

#include "tensor.h"
#include "utils.h"

namespace ops {
void matmul(const Tensor& A, const Tensor& B, Tensor& OUT);
void RMSNorm(const Tensor& IN, const Tensor& W, Tensor& OUT, float eps = 1e-5f);
void SiLU(const Tensor& IN, Tensor& OUT);
void add(const Tensor& A, const Tensor& B, Tensor& OUT);
void mul(const Tensor& A, const Tensor& B, Tensor& OUT);
void softmax(Tensor& INOUT);
void embed(const std::vector<int>& ids, const Tensor& We, Tensor& OUT);
void RoPE(Tensor& Q, Tensor& K, const int head_dim, const int start_pos, const RopeCache& rc);
void attn(const Tensor& Q,
          const Tensor& K,
          const Tensor& V,
          Tensor&       OUT,
          const int     n_heads,
          const int     n_kv_heads,
          const int     head_dim);
}  // namespace ops
