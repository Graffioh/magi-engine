#include "ops.h"

#include <stdexcept>

// out = input^T
Tensor transpose(const Tensor& input) {
    Tensor output;
    output.rows = input.cols;
    output.cols = input.rows;
    output.data.resize(output.rows * output.cols);

    for (int i = 0; i < input.rows; ++i) {
        for (int j = 0; j < input.cols; ++j) {
            output(j, i) = input(i, j);
        }
    }

    return output;
}

// out = A ∙ B
void matmul(const Tensor& A, const Tensor& B, Tensor& OUT) {
    if (A.cols != B.rows) {
        throw std::invalid_argument("Incompatible dimensions for matrix multiplication");
    }
    OUT.rows = A.rows;
    OUT.cols = B.cols;
    OUT.data.resize(OUT.rows * OUT.cols, 0.0f);

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

// out = A ∙ B^T
// A: (M, K), B: (N, K)  ->  OUT: (M, N)
void matmul_nt(const Tensor& A, const Tensor& B, Tensor& OUT) {
    if (A.cols != B.cols) {
        throw std::invalid_argument("Incompatible dimensions for matmul_nt (A.cols must equal B.cols)");
    }
    OUT.rows = A.rows;
    OUT.cols = B.rows;
    OUT.data.resize(OUT.rows * OUT.cols, 0.0f);

    for (int i = 0; i < OUT.rows; ++i) {
        for (int j = 0; j < OUT.cols; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < A.cols; ++k) {
                sum += A(i, k) * B(j, k);
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

// out = A * B (element-wise)
void mul(const Tensor& A, const Tensor& B, Tensor& OUT) {
    if (A.rows != B.rows || A.cols != B.cols) {
        throw std::invalid_argument("Incompatible dimensions for element-wise multiplication");
    }
    OUT.rows = A.rows;
    OUT.cols = A.cols;
    OUT.data.resize(OUT.rows * OUT.cols);

    for (int i = 0; i < OUT.rows; ++i) {
        for (int j = 0; j < OUT.cols; ++j) {
            OUT(i, j) = A(i, j) * B(i, j);
        }
    }
}

// IN(i, j) = IN(i, j) / rms(IN[i, :]) * weight[j]
void RMSNorm(Tensor& IN, const std::vector<float>& weight, float epsilon) {
    if (static_cast<int>(weight.size()) != IN.cols) {
        throw std::invalid_argument("RMSNorm weight size must match IN.cols");
    }

    for (int i = 0; i < IN.rows; ++i) {
        float sum_squares = 0.0f;
        for (int j = 0; j < IN.cols; ++j) {
            const float v = IN(i, j);
            sum_squares += v * v;
        }

        const float mean_squares = sum_squares / IN.cols;
        const float inv_rms = 1.0f / std::sqrt(mean_squares + epsilon);

        for (int j = 0; j < IN.cols; ++j) {
            IN(i, j) = IN(i, j) * inv_rms * weight[j];
        }
    }
}

// out = SiLU(IN) = IN / (1 + exp(-IN))
void SiLU(Tensor& IN) {
    for (int i = 0; i < IN.rows * IN.cols; ++i) {
        float x = IN.data[i];
        IN.data[i] = x / (1.0f + std::exp(-x));
    }
}

// out = softmax(IN) applied row-wise
void softmax(Tensor& IN) {
    for (int i = 0; i < IN.rows; ++i) {
        // Find the max value in the row for numerical stability
        float max_val = -std::numeric_limits<float>::infinity();
        for (int j = 0; j < IN.cols; ++j) {
            if (IN(i, j) > max_val) {
                max_val = IN(i, j);
            }
        }

        // Compute the sum of exp(x - max_val) for the row
        float sum_exp = 0.0f;
        for (int j = 0; j < IN.cols; ++j) {
            sum_exp += std::exp(IN(i, j) - max_val);
        }

        // Normalize the row by dividing exp(x - max_val) by the sum of exp
        for (int j = 0; j < IN.cols; ++j) {
            IN(i, j) = std::exp(IN(i, j) - max_val) / sum_exp;
        }
    }
}

// IN = IN * factor
void scale(Tensor& IN, float factor) {
    for (int i = 0; i < IN.rows * IN.cols; ++i) {
        IN.data[i] *= factor;
    }
}
