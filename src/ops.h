#pragma once

#include "tensor.h"

void mattranspose(const Tensor& input, Tensor& output);
void matmul(const Tensor& A, const Tensor& B, Tensor& OUT);
void matvec(const Tensor& A, const std::vector<float>& x, Tensor& OUT);
void add(const Tensor& A, const Tensor& B, Tensor& OUT);
void RMSNorm(Tensor& IN);
void matmul_bad_locality(const Tensor& A, const Tensor& B, Tensor& OUT);
