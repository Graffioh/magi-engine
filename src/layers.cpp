#include "layers.h"

#include "ops.h"

#include <assert.h>

LinearLayer::LinearLayer(Tensor W) : W_(std::move(W)) {}

void LinearLayer::forward(const Tensor& IN, Tensor& OUT) const {
    ops::matmul(IN, W_, OUT);
}

EmbeddingLayer::EmbeddingLayer(Tensor We) : We_(std::move(We)) {}

void EmbeddingLayer::forward(const std::vector<int>& token_ids, Tensor& OUT) const {
    ops::embed(token_ids, We_, OUT);
}

MLP::MLP(LinearLayer gate, LinearLayer up, LinearLayer down) :
    gate_(std::move(gate)),
    up_(std::move(up)),
    down_(std::move(down)) {}

void MLP::forward(const Tensor& IN, Tensor& OUT) const {
    assert(IN.rank() == 2);  // CPU inference engine => batch = 1 during computation (since it's sequential)
    std::vector<int> tmp_shape = { IN.dim(0), gate_.get_weight_out_features() };

    // GATE = IN @ W_g^T
    Tensor GATE(tmp_shape);
    gate_.forward(IN, GATE);

    // GATE = IN @ W_u^T
    Tensor UP(tmp_shape);
    up_.forward(IN, UP);

    ops::SiLU(GATE, GATE);

    // GATE = SiLU(GATE) ° UP
    ops::mul(GATE, UP, GATE);

    // OUT = GATE @ W_d^T
    down_.forward(GATE, OUT);
}

AttentionLayer::AttentionLayer(const ModelConfig& config,
                               LinearLayer        q_proj,
                               LinearLayer        k_proj,
                               LinearLayer        v_proj,
                               LinearLayer        out_proj) :
    q_proj_(std::move(q_proj)),
    k_proj_(std::move(k_proj)),
    v_proj_(std::move(v_proj)),
    out_proj_(std::move(out_proj)),
    head_dim_(config.head_dim),
    n_heads_(config.n_heads),
    n_kv_heads_(config.n_kv_heads) {}

void AttentionLayer::forward(const Tensor& IN, Tensor& OUT, const int start_pos, const RopeCache& rc) const {
    const int T = IN.dim(0);
    Tensor    Q({ T, n_heads_ * head_dim_ });
    Tensor    K({ T, n_kv_heads_ * head_dim_ });
    Tensor    V({ T, n_kv_heads_ * head_dim_ });

    q_proj_.forward(IN, Q);
    k_proj_.forward(IN, K);
    v_proj_.forward(IN, V);

    ops::RoPE(Q, K, head_dim_, start_pos, rc);

    Tensor ATTN({ T, n_heads_ * head_dim_ });
    ops::attn(Q, K, V, ATTN, n_heads_, n_kv_heads_, head_dim_);
    out_proj_.forward(ATTN, OUT);
}
