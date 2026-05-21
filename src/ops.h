#include "tensor.h"

class Operations {
    private:
        void matmul_2d(const float* A_data, const float* B_data, float* OUT_data, int M, int K, int N);
    public:
        void matmul(const Tensor& A, const Tensor& B, Tensor& OUT);
        void RMSNorm(const Tensor& IN, const Tensor& W, Tensor& OUT, float eps = 1e-5);
        void SiLU(const Tensor& IN, Tensor& OUT);
};