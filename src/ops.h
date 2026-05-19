#include "tensor.h"

class Operations {
    private:
        void matmul_2d(const float* A_data, const float* B_data, float* OUT_data, int M, int K, int N);
    public:
        void matmul(const Tensor& A, const Tensor& B, Tensor& OUT);
};