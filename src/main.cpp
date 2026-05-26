#include "layers.h"
#include "ops.h"
#include "tensor.h"

#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

static void print_shape(const Tensor &t) {
  std::cout << "  shape   = [";
  for (int i = 0; i < t.rank(); ++i) {
    std::cout << t.shape()[i];
    if (i + 1 < t.rank())
      std::cout << ", ";
  }
  std::cout << "]\n";

  std::cout << "  strides = [";
  for (int i = 0; i < t.rank(); ++i) {
    std::cout << t.strides()[i];
    if (i + 1 < t.rank())
      std::cout << ", ";
  }
  std::cout << "]\n";
}

static void print_data(const Tensor &t) {
  const float *p = t.data_ptr();
  int n = t.num_elements();
  std::cout << "  data    = [";
  for (int i = 0; i < n; ++i)
    std::cout << p[i] << (i + 1 < n ? ", " : "");
  std::cout << "]\n";
}

static void fill(Tensor &t, const std::vector<float> &vals) {
  float *p = t.data_ptr();
  for (size_t i = 0; i < vals.size(); ++i)
    p[i] = vals[i];
}

static void fill_random(Tensor &t, uint32_t seed = 42, float lo = -0.1f, float hi = 0.1f) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(lo, hi);
  float *p = t.data_ptr();
  int n = t.num_elements();
  for (int i = 0; i < n; ++i)
    p[i] = dist(rng);
}

int main() {
  // === matmul ===
  Tensor A({2, 2}), B({3, 2}), OUT({2, 3});
  fill(A, {1, 2, 3, 4});
  fill(B, {5, 8, 6, 9, 7, 10});
  ops::matmul(A, B, OUT);

  std::cout << "matmul (2,2) x (2,3):\n";
  print_shape(OUT);
  print_data(OUT);

  // === SwiGLU MLP ===
  Tensor Wg({4, 2}); // (intermediate_size, hidden_size)
  Tensor Wu({4, 2}); // ""
  Tensor Wd({2, 4}); // (hidden_size, intermediate_size)

  fill_random(Wg);
  fill_random(Wu);
  fill_random(Wd);

  LinearLayer gate(Wg);
  LinearLayer up(Wu);
  LinearLayer down(Wd);

  MLP swiglu_mlp(gate, up, down);
}
