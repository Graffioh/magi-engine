# Roadmap

Goal: a CPU-only C++ inference engine that runs TinyLlama
(`TinyLlama-1.1B-Chat-v1.0`) end-to-end, from a tokenized prompt to streamed
output tokens. The roadmap is staged so each milestone produces something
runnable and verifiable on its own.

Target model reference: TinyLlama 1.1B (22 layers, hidden 2048, 32 query heads,
4 KV heads, head_dim 64, intermediate 5632, vocab 32000, RoPE base 10000,
RMSNorm, SwiGLU MLP, grouped-query attention).

## Stage 0 — Foundations

Scope: storage, math primitives, and a build that runs.

- [x] `Tensor`: contiguous float storage, shape, strides, indexing helpers.
- [ ] Shape validation utilities that throw with useful messages.
- [ ] `ops`: `matmul`, `matmul_transposed_b`, `rmsnorm`, `softmax`, `silu`,
      elementwise add/mul.
- [ ] `main.cpp` smoke test: build a tiny tensor, run an op, print a value.

Exit criteria: `cmake --build build && ./build/magi_engine` prints expected
numbers for a hand-computed matmul + RMSNorm.

## Stage 1 — Transformer Building Blocks

Scope: every layer TinyLlama needs, tested in isolation.

- [ ] `LinearLayer` (no bias, row-major weights).
- [ ] Token embedding lookup.
- [ ] RoPE: precomputed `inv_freq`, applied to Q and K per head.
- [ ] Grouped-query causal self-attention (32 Q heads → 4 KV heads, head repeat).
- [ ] SwiGLU gated MLP (`silu(gate(x)) * up(x)` → `down`).
- [ ] Pre-norm transformer block: `x + attn(rmsnorm(x))`, `x + mlp(rmsnorm(x))`.
- [ ] Final RMSNorm + LM head projection to vocab logits.

Exit criteria: a single transformer block forward pass with random weights runs
without shape errors and produces stable (non-NaN) outputs.

## Stage 2 — Weight Loading

Scope: load real TinyLlama weights into the layers above.

- [ ] Pick a format. Recommended: `safetensors` (simple header + raw fp32/fp16
      tensors). Alternative: a tiny custom dump produced by a Python script.
- [ ] `weights.*`: parse header, mmap or read tensor blobs, expose by name.
- [ ] Name mapping: HF parameter names → engine layers
      (`model.layers.{i}.self_attn.q_proj.weight`, etc.).
- [ ] Dtype: start fp32. Convert from fp16 on load if needed.
- [ ] Sanity check: norm of embeddings and a known weight matches a Python
      reference.

Exit criteria: model constructed from a real checkpoint; one forward pass on a
fixed input matches reference logits within a small tolerance.

## Stage 3 — Tokenizer

Scope: turn strings into token ids and back.

- [ ] Choose path:
  - Minimal: load TinyLlama's `tokenizer.model` (SentencePiece BPE) via a
    vendored small SentencePiece reader, or
  - Simpler: pre-encode prompts in Python and feed token ids to the engine
    while the C++ tokenizer matures.
- [ ] Implement encode/decode for the chat template used by TinyLlama-Chat
      (`<|system|>`, `<|user|>`, `<|assistant|>`).
- [ ] Round-trip test on a known string.

Exit criteria: `encode(decode(ids)) == ids` for a handful of prompts; chat
template produces the same ids as the HF tokenizer for a fixed example.

## Stage 4 — End-to-End Forward Pass

Scope: full model, one token at a time, no cache yet.

- [ ] `Model::forward(tokens) -> logits`: embedding → N blocks → final norm →
      LM head.
- [ ] Greedy argmax sampling.
- [ ] Run a short prompt and decode the next ~20 tokens.

Exit criteria: greedy continuation of a fixed prompt matches a Python reference
(allow small numerical drift after several tokens).

## Stage 5 — KV Cache And Autoregressive Decoding

Scope: make generation practical.

- [ ] Per-layer KV cache: shape `[num_kv_heads, max_seq, head_dim]`.
- [ ] Prefill path: process the whole prompt, fill the cache.
- [ ] Decode path: feed one token, append one row to K/V, attend over cache.
- [ ] Position counter shared by RoPE and the cache writer.

Exit criteria: prefill + decode produces the same tokens as Stage 4 but at much
higher tokens/second, and memory is bounded by `max_seq`.

## Stage 6 — Sampling Controls

Scope: useful generation knobs.

- [ ] Temperature.
- [ ] Top-k.
- [ ] Top-p (nucleus).
- [ ] Repetition penalty (optional).
- [ ] Deterministic seed.

Exit criteria: same seed + same params → identical output across runs.

## Stage 7 — CPU Performance (Optional, Last)

Scope: make it fast without hiding the math.

- [ ] Cache-friendly matmul (blocked / loop-reordered).
- [ ] OpenMP parallelism on the outer matmul loop.
- [ ] Fused RMSNorm + projection where it stays readable.
- [ ] Optional: int8 / Q4 weight quantization for the linear layers.
- [ ] Lightweight benchmark harness reporting tokens/sec for prefill and decode.

Exit criteria: measurable speedup on prefill and decode with unchanged outputs
(within tolerance) and code that still reads like a teaching implementation.

## Out Of Scope

- GPU / Metal / CUDA backends.
- Training, fine-tuning, LoRA.
- Multi-model orchestration, server/HTTP layer.
- Speculative decoding, batching across requests.

## References

- TinyLlama: Zhang et al., *TinyLlama: An Open-Source Small Language Model*
  (2024).
- LLaMA: Touvron et al., *LLaMA: Open and Efficient Foundation Language Models*
  (2023).
- RoPE: Su et al., *RoFormer: Enhanced Transformer with Rotary Position
  Embedding* (2021).
- RMSNorm: Zhang & Sennrich, *Root Mean Square Layer Normalization* (2019).
- GQA: Ainslie et al., *GQA: Training Generalized Multi-Query Transformer
  Models from Multi-Head Checkpoints* (2023).
- SwiGLU: Shazeer, *GLU Variants Improve Transformer* (2020).
