#include "ops.h"

#include <stdexcept>

void mattranspose(const Tensor& input, Tensor& output) {
    output.rows = input.cols;
    output.cols = input.rows;
    output.data.resize(output.rows * output.cols);

    for (int i = 0; i < input.rows; ++i) {
        for (int j = 0; j < input.cols; ++j) {
            output(j, i) = input(i, j);
        }
    }
}

void matmul(const Tensor& A, const Tensor& B, Tensor& OUT) {
    if (A.cols != B.rows) {
        throw std::invalid_argument("Incompatible dimensions for matrix multiplication");
    }
    OUT.rows = A.rows;
    OUT.cols = B.cols;
    OUT.data.resize(OUT.rows * OUT.cols, 0.0f);

    Tensor BT;
    mattranspose(B, BT);

    // Both A(i, k) and BT(j, k) are read row-wise as k increases.
    for (int i = 0; i < OUT.rows; ++i) {
        for (int j = 0; j < OUT.cols; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < A.cols; ++k) {
                sum += A(i, k) * BT(j, k);
            }
            OUT(i, j) = sum;
        }
    }
}

void matmul_bad_locality(const Tensor& A, const Tensor& B, Tensor& OUT) {
    if (A.cols != B.rows) {
        throw std::invalid_argument("Incompatible dimensions for matrix multiplication");
    }
    OUT.rows = A.rows;
    OUT.cols = B.cols;
    OUT.data.resize(OUT.rows * OUT.cols, 0.0f);

    // Baseline: OUT is written row-wise, but B(k, j) jumps across rows.
    for (int i = 0; i < OUT.rows; ++i) {
        for (int j = 0; j < OUT.cols; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < A.cols; ++k) {
                sum += A(i, k) * B(k, j);
            }
            OUT(i, j) = sum;
        }
    }
}
