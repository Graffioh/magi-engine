#include "tensor.h"

class Operations {
    private:
        void matmul_2d(const float* A_data, const float* B_data, float* OUT_data, int M, int K, int N);
        void RMSNorm_1d(const float* IN_data, const float* W_data, float* OUT_data, int dim, float eps);
    public:
        void matmul(const Tensor& A, const Tensor& B, Tensor& OUT);
        void RMSNorm(const Tensor& IN, const Tensor& W, Tensor& OUT, float eps = 1e-5);
        void SiLU(const Tensor& IN, Tensor& OUT);
        void add(const Tensor& A, const Tensor& B, Tensor& OUT);
        void mul(const Tensor& A, const Tensor& B, Tensor& OUT);
};