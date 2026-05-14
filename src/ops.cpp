#include "ops.h"

#include <stdexcept>

// out = input^T
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

// out = A ∙ B
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

// out = weight * x
void matvec(const Tensor& A, const std::vector<float>& x, Tensor& OUT) {
    if (A.cols != x.size()) {
        throw std::invalid_argument("Incompatible dimensions for matrix-vector multiplication");
    }
    OUT.rows = A.rows;
    OUT.cols = 1;
    OUT.data.resize(OUT.rows, 0.0f);

    for (int i = 0; i < OUT.rows; ++i) {
        float sum = 0.0f;
        for (int j = 0; j < A.cols; ++j) {
            sum += A(i, j) * x[j];
        }
        OUT(i, 0) = sum;
    }
}

// out = A + B
void add(const Tensor& A, const Tensor& B, Tensor& OUT) {
    if (A.rows != B.rows || A.cols != B.cols) {
        throw std::invalid_argument("Incompatible dimensions for matrix addition");
    }
    OUT.rows = A.rows;
    OUT.cols = A.cols;
    OUT.data.resize(OUT.rows * OUT.cols);

    for (int i = 0; i < OUT.rows; ++i) {
        for (int j = 0; j < OUT.cols; ++j) {
            OUT(i, j) = A(i, j) + B(i, j);
        }
    }
}

void RMSNorm(Tensor& IN) {
    float sum_squares = 0.0f;
    int count = IN.rows * IN.cols;

    for (int i = 0; i < count; ++i) {
        sum_squares += IN.data[i] * IN.data[i];
    }

    // compute rms
    const float mean_squares = sum_squares / count;
    const float epsilon = 1e-8f; // Small constant to prevent division by zero
    const float rms = std::sqrt(mean_squares + epsilon);

    // use rms to normalize the input tensor
    for (int i = 0; i < count; ++i) {
        IN.data[i] /= rms;
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
