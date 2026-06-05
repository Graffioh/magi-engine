#include "gguf_model.h"

#include "layers.h"
#include "tensor.h"
#include "utils.h"

#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

// Move the named tensor view OUT of g.tensors. The view keeps the mmap alive via
// its owner_, so it stays valid after `g` is destroyed. Throws a clear error if
// the name is absent -- a malformed/incomplete checkpoint should fail loudly here
// rather than later as a missing-weight crash.
Tensor take_tensor(gguf::GGUF& g, const std::string& name) {
    auto it = g.tensors.find(name);
    if (it == g.tensors.end()) {
        throw std::runtime_error("gguf_model: missing required tensor: " + name);
    }
    return std::move(it->second);
}

// Make a SECOND read-only view over the same tensor data without moving or
// copying it. Tensor is move-only and a given view can only be moved out of the
// map once, but a tied lm_head needs to reference token_embd.weight while the
// EmbeddingLayer also holds it. We rebuild a view from the original's shape +
// data pointer, co-owning the same mmap (g.mapping) so both views keep the file
// mapped. No tensor data is copied -- the two views alias the same bytes.
Tensor clone_view(const Tensor& src, std::shared_ptr<void> owner) {
    return Tensor(src.shape(), src.data_ptr(), std::move(owner));
}

}  // namespace

ModelConfig config_from_gguf(const gguf::GGUF& g) {
    ModelConfig cfg;

    cfg.n_layers          = static_cast<int>(g.u32("llama.block_count"));
    cfg.hidden_size       = static_cast<int>(g.u32("llama.embedding_length"));
    cfg.intermediate_size = static_cast<int>(g.u32("llama.feed_forward_length"));
    cfg.n_heads           = static_cast<int>(g.u32("llama.attention.head_count"));
    cfg.n_kv_heads        = static_cast<int>(g.u32("llama.attention.head_count_kv"));

    // head_dim is usually carried explicitly as the RoPE dimension count; if it
    // is absent fall back to the standard hidden_size / n_heads split.
    cfg.head_dim = g.has("llama.rope.dimension_count")
                       ? static_cast<int>(g.u32("llama.rope.dimension_count"))
                       : cfg.hidden_size / cfg.n_heads;

    cfg.max_seq_len = static_cast<int>(g.u32("llama.context_length"));
    cfg.rms_eps     = g.f32("llama.attention.layer_norm_rms_epsilon");

    // rope.freq_base is stored as a float in GGUF; our config keeps it as an int
    // (the RopeCache base is integral). Default to 10000 when omitted.
    cfg.rope_base = g.has("llama.rope.freq_base")
                        ? static_cast<int>(g.f32("llama.rope.freq_base"))
                        : 10000;

    // vocab_size: prefer the explicit kv; else the tokenizer token list length;
    // else recover it from the embedding table's row count (shape {vocab, hidden}).
    if (g.has("llama.vocab_size")) {
        cfg.vocab_size = static_cast<int>(g.u32("llama.vocab_size"));
    } else if (g.has("tokenizer.ggml.tokens")) {
        cfg.vocab_size = static_cast<int>(g.str_array("tokenizer.ggml.tokens").size());
    } else {
        cfg.vocab_size = g.tensor("token_embd.weight").dim(0);
    }

    return cfg;
}

Model build_model_from_gguf(gguf::GGUF&& g) {
    const ModelConfig cfg = config_from_gguf(g);

    // Embedding table: {vocab, hidden}. Take the view once; if the lm_head is
    // tied we make a second aliasing view of these same bytes below.
    Tensor token_embd = take_tensor(g, "token_embd.weight");

    // lm_head: an explicit output.weight if present, otherwise TIE to the
    // embedding table (TinyLlama and friends omit output.weight). Tying means two
    // layers must view the same data -- but a view can only be moved out of the
    // map once, so for the tied case we build a separate aliasing view via
    // clone_view (no data copy; both views co-own the mmap).
    Tensor lm_head_w = g.tensors.count("output.weight")
                           ? take_tensor(g, "output.weight")
                           : clone_view(token_embd, g.mapping);

    EmbeddingLayer embed(std::move(token_embd));
    LinearLayer    lm_head(std::move(lm_head_w));

    RMSNormLayer final_norm(take_tensor(g, "output_norm.weight"), cfg.rms_eps);

    std::vector<TransformerBlock> blocks;
    blocks.reserve(cfg.n_layers);
    for (int i = 0; i < cfg.n_layers; ++i) {
        const std::string p = "blk." + std::to_string(i) + ".";

        AttentionLayer attn(cfg,
                            LinearLayer(take_tensor(g, p + "attn_q.weight")),
                            LinearLayer(take_tensor(g, p + "attn_k.weight")),
                            LinearLayer(take_tensor(g, p + "attn_v.weight")),
                            LinearLayer(take_tensor(g, p + "attn_output.weight")));

        MLP mlp(LinearLayer(take_tensor(g, p + "ffn_gate.weight")),
                LinearLayer(take_tensor(g, p + "ffn_up.weight")),
                LinearLayer(take_tensor(g, p + "ffn_down.weight")));

        RMSNormLayer attn_norm(take_tensor(g, p + "attn_norm.weight"), cfg.rms_eps);
        RMSNormLayer ffn_norm(take_tensor(g, p + "ffn_norm.weight"), cfg.rms_eps);

        blocks.push_back(TransformerBlock(std::move(attn), std::move(mlp),
                                          std::move(attn_norm), std::move(ffn_norm)));
    }

    RopeCache rc(cfg.max_seq_len, cfg.head_dim, cfg.rope_base);

    return Model(std::move(embed), std::move(blocks), std::move(final_norm),
                 std::move(lm_head), std::move(rc), cfg);
}
