#pragma once

#include "tensor.h"

Tensor transpose(const Tensor& input);
void matmul(const Tensor& A, const Tensor& B, Tensor& OUT);
void matmul_nt(const Tensor& A, const Tensor& B, Tensor& OUT);
void matvec(const Tensor& A, const std::vector<float>& x, Tensor& OUT);
void add(const Tensor& A, const Tensor& B, Tensor& OUT);
void mul(const Tensor& A, const Tensor& B, Tensor& OUT);
void RMSNorm(Tensor& IN, const std::vector<float>& weight, float epsilon = 1e-5f);
void SiLU(Tensor& IN);
void softmax(Tensor& IN);
void scale(Tensor& IN, float factor);
