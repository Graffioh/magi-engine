#include "layers.h"

#include "ops.h"

LinearLayer::LinearLayer(int input_dim, int output_dim, bool use_bias) {
  W.rows = output_dim;
  W.cols = input_dim;
  W.data.resize(output_dim * input_dim, 0.0f);

  bias.resize(use_bias ? output_dim : 0, 0.0f);
}

void LinearLayer::forward(const Tensor &IN, Tensor &OUT) const {
  matmul_nt(IN, W, OUT); // (seq_len, input_dim) x (output_dim, input_dim)^T -> (seq_len, output_dim)

  if (has_bias()) {
    for (int i = 0; i < OUT.rows; ++i) {
      for (int j = 0; j < OUT.cols; ++j) {
        OUT(i, j) += bias[j];
      }
    }
  }
}

GatedMLP::GatedMLP(int hidden_size, int intermediate_size)
: gate_proj(hidden_size, intermediate_size),
  up_proj(hidden_size, intermediate_size),
  down_proj(intermediate_size, hidden_size) {}

void GatedMLP::forward(const Tensor& IN, Tensor& OUT) const {
  Tensor gate_out; // (seq_len, intermediate_size)
  gate_proj.forward(IN, gate_out); 

  Tensor up_out; // (seq_len, intermediate_size)
  up_proj.forward(IN, up_out); 

  SiLU(gate_out);

  Tensor hidden; // (seq_len, intermediate_size)
  mul(gate_out, up_out, hidden); // Element-wise multiplication for gating

  down_proj.forward(hidden, OUT);
}
