#include "ops.h"

#include <chrono>
#include <iostream>

namespace {

Tensor make_tensor(int rows, int cols) {
  Tensor tensor(rows, cols);

  for (int i = 0; i < rows * cols; ++i) {
    tensor.data[i] = static_cast<float>((i % 17) - 8) / 17.0f;
  }

  return tensor;
}

double time_matmul(void (*fn)(const Tensor&, const Tensor&, Tensor&), const Tensor& A,
                   const Tensor& B, Tensor& OUT, int runs) {
  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < runs; ++i) {
    fn(A, B, OUT);
  }

  auto end = std::chrono::steady_clock::now();
  std::chrono::duration<double, std::milli> elapsed = end - start;
  return elapsed.count() / runs;
}

} // namespace

int main() {
  constexpr int size = 512;
  constexpr int runs = 5;

  Tensor A = make_tensor(size, size);
  Tensor B = make_tensor(size, size);
  Tensor OUT;

  matmul(A, B, OUT);

  double matmul_ms = time_matmul(matmul, A, B, OUT, runs);

  std::cout << "matrix size: " << size << " x " << size << '\n';
  std::cout << "runs: " << runs << '\n';
  std::cout << "matmul time: " << matmul_ms << " ms/run" << '\n';

  return 0;
}
