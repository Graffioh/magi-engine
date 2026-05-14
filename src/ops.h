#pragma once

#include "tensor.h"

void mattranspose(const Tensor& input, Tensor& output);
void matmul(const Tensor& A, const Tensor& B, Tensor& OUT);
void matmul_bad_locality(const Tensor& A, const Tensor& B, Tensor& OUT);
