#include <cmath>
#include <stdexcept>

#include "layers.h"
#include "ops.h"

LinearLayer::LinearLayer(int input_dim, int output_dim, bool use_bias)
: W(output_dim, input_dim) {
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

  Tensor up_out; // ""
  up_proj.forward(IN, up_out); 

  SiLU(gate_out);

  Tensor hidden; // ""
  mul(gate_out, up_out, hidden); 

  down_proj.forward(hidden, OUT);
}

Embedding::Embedding(int vocab_size, int embedding_size) : W(vocab_size, embedding_size) {}

void Embedding::forward(const std::vector<int>& token_ids, Tensor& OUT) const {
  OUT = Tensor(token_ids.size(), W.cols); // (seq_len, embedding_dim)

  for (size_t i = 0; i < token_ids.size(); ++i) {
    int token_id = token_ids[i];
    if (token_id < 0 || token_id >= W.rows) {
      throw std::out_of_range("Token ID out of range in Embedding forward");
    }
    for (int j = 0; j < W.cols; ++j) {
      OUT(i, j) = W(token_id, j);
    }
  }
}

SelfAttention::SelfAttention(
  int hidden_size,
  int num_heads,
  int num_kv_heads
)
  : q_proj(hidden_size, hidden_size, false),
    k_proj(hidden_size, num_kv_heads * (hidden_size / num_heads), false),
    v_proj(hidden_size, num_kv_heads * (hidden_size / num_heads), false),
    o_proj(hidden_size, hidden_size, false),
    hidden_size(hidden_size),
    num_heads(num_heads),
    num_kv_heads(num_kv_heads),
    head_dim(hidden_size / num_heads) {
  if (hidden_size % num_heads != 0) {
    throw std::invalid_argument("hidden_size must be divisible by num_heads");
  }

  if (num_heads % num_kv_heads != 0) {
    throw std::invalid_argument("num_heads must be divisible by num_kv_heads");
  }
}

// upper triangle masking to prevent attending to future tokens
void SelfAttention::causal_mask(Tensor& attn_scores) const {
  for (int i = 0; i < attn_scores.rows; ++i) {
    for (int j = i + 1; j < attn_scores.cols; ++j) {
      attn_scores(i, j) = -std::numeric_limits<float>::infinity();
    }
  }
}

void SelfAttention::attn( const Tensor& q, const Tensor& k, const Tensor& v, Tensor& OUT) const {
  Tensor scores;

  // scores = q @ k^T
  matmul_nt(q, k, scores);

  // scores = scores / sqrt(head_dim)
  scale(scores, 1.0f / std::sqrt(static_cast<float>(head_dim)));

  // prevent token i from attending to future token j > i
  causal_mask(scores);

  // softmax each row
  softmax(scores);

  // OUT = scores @ v
  matmul(scores, v, OUT);
}

void SelfAttention::apply_rope(Tensor& x, int start_pos, float theta) const {
  int seq_len = x.rows;

  if (x.cols % head_dim != 0) {
    throw std::invalid_argument("RoPE input cols must be divisible by head_dim");
  }

  int heads = x.cols / head_dim;

  for (int pos = 0; pos < seq_len; ++pos) {
    int absolute_pos = start_pos + pos;

    for (int h = 0; h < heads; ++h) {
      int head_offset = h * head_dim;

      for (int i = 0; i < head_dim; i += 2) {
        float inv_freq = 1.0f / std::pow(
          theta,
          static_cast<float>(i) / static_cast<float>(head_dim)
        );

        float angle = static_cast<float>(absolute_pos) * inv_freq;
        float c = std::cos(angle);
        float s = std::sin(angle);

        int idx0 = head_offset + i;
        int idx1 = head_offset + i + 1;

        float x0 = x(pos, idx0);
        float x1 = x(pos, idx1);

        x(pos, idx0) = x0 * c - x1 * s;
        x(pos, idx1) = x0 * s + x1 * c;
      }
    }
  }
}

void SelfAttention::forward(const Tensor& IN, Tensor& OUT) const {
  Tensor q, k, v;
  q_proj.forward(IN, q);
  k_proj.forward(IN, k);
  v_proj.forward(IN, v);

  apply_rope(q, 0);
  apply_rope(k, 0);

  Tensor attn_out;
  attn(q, k, v, attn_out);

  o_proj.forward(attn_out, OUT);
}
