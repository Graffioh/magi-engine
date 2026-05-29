# Roadmap

Goal: a CPU-only C++ inference engine that runs TinyLlama
(`TinyLlama-1.1B-Chat-v1.0`) end-to-end, from a tokenized prompt to streamed
output tokens. The roadmap is staged so each milestone produces something
runnable and verifiable on its own.

Target model reference: TinyLlama 1.1B (22 layers, hidden 2048, 32 query heads,
4 KV heads, head_dim 64, intermediate 5632, vocab 32000, RoPE base 10000,
RMSNorm, SwiGLU MLP, grouped-query attention).

The big structural idea: **weight loading and a toy reference model come
before attention.** Attention is the hardest piece to get right and
near-impossible to debug without a golden reference. We build the loading
infrastructure + a tiny dummy model + Python-computed golden outputs first,
then add attention against that scaffolding.

> **Progress note (out-of-order work).** Stage 2's *infrastructure* (the
> `Tensor` storage refactor and the `MappedFile` mmap wrapper) was built
> early, before the rest of Stage 1, because both were small and unblocked.
> What still remains in Stage 2 is the part that depends on a file format:
> the custom binary layout and `load_weights()`. Remaining near-term work in
> roadmap order: **token embedding lookup (Stage 1)** → **binary format +
> `load_weights` (Stage 2)** → **dummy weights + goldens (Stage 3)**.

## Stage 0 — Foundations

Scope: storage, math primitives, and a build that runs.

- [x] `Tensor`: contiguous float storage, shape, strides, indexing helpers.
- [x] Asserts on op preconditions (shape compatibility, rank, etc.).
- [x] `ops`: `matmul` (B in transposed `(N, K)` storage), `RMSNorm`, `softmax`,
      `SiLU`, elementwise `add` / `mul`.
- [x] `main.cpp` scratchpad: build a tiny tensor, run an op, print a value.
- [x] `tests/` harness with hand-computed expected outputs.

Exit criteria: `cmake --build build && ./build/magi_tests` passes; the
scratchpad prints expected numbers for a hand-computed matmul.

## Stage 1 — Simple Building Blocks

Scope: the layers that don't need attention. These can be built and tested
against random inputs without any weight loading infrastructure.

- [x] `LinearLayer` (no bias, row-major weights in `(out, in)` storage).
- [x] SwiGLU `MLP` (`down(SiLU(gate(x)) * up(x))`), with scratch buffers
      allocated inside `forward()` for now.
- [ ] Token embedding lookup (`out = embed_weights[token_id]`).
- [x] Hand-computed tests for `MLP` in `tests/`.
- [ ] Hand-computed test for embedding in `tests/`.

Exit criteria: scratchpad constructs an `MLP` from random tensors, runs
`forward(x)` on a random input, prints a finite, correctly-shaped output.

## Stage 2 — Weight Loading Infrastructure

Scope: replace the "fill weights from a `std::vector` in code" pattern with
real weight loading via `mmap`. This is foundational for everything that
follows.

- [x] `Tensor` refactor: backing storage becomes
      `float* + std::shared_ptr<void>` so a tensor can either own
      its data (vector-backed) or view into an mmap'd region. Existing tests
      keep passing unchanged.
- [x] `MappedFile` RAII wrapper (`open` → `fstat` → `mmap(PROT_READ)` →
      `munmap`/`close` in destructor). POSIX-only. (Has its own smoke test in
      `tests/test_mmap.cpp`; named `MappedFile` rather than `MmapFile`.)
- [ ] Custom binary file format (magic, version, directory of name + rank +
      shape + offset, 16-byte-aligned float data section).
- [ ] `load_weights(path)` returning `std::unordered_map<std::string, Tensor>`,
      with each tensor's data pointer aiming into the mmap and the
      `shared_ptr<MmapFile>` keeping the region alive.
- [ ] Single-tensor smoke test: write a `.bin`, load it back, verify byte-for-
      byte.

Exit criteria: a binary file written by a small C++ writer (or Python) can be
loaded via `load_weights` and the tensors printed — values match what was
written.

## Stage 3 — Dummy Model And Golden Outputs

Scope: a toy model file in the same format used for real weights, plus
Python-computed reference outputs for every op. This becomes the ground truth
for the rest of development.

- [ ] Toy dims: `vocab=32`, `hidden=8`, `intermediate=16`, `n_layers=2`,
      `n_heads=2`, `n_kv_heads=1`, `head_dim=4`, `max_seq=16`. Small enough
      to print whole tensors, structured enough to exercise GQA and stacking.
- [ ] `tools/make_dummy_weights.py`: NumPy-based, fixed seed, emits the full
      Llama-style weight set with HuggingFace-style names
      (`model.layers.{i}.self_attn.q_proj.weight`, etc.).
- [ ] `tools/make_golden_outputs.py`: same dummy weights, reference forward
      pass in NumPy (or stub PyTorch); emits a sibling file with expected
      outputs for embedding, RMSNorm, MLP, and (later) RoPE / attention.
- [ ] Scratchpad wire-up: load `toy.bin`, build one `MLP` from layer-0
      weights, run on `embed(token_id=5)`, print and eyeball.
- [ ] Tests load the toy weights and check `MLP` / embedding outputs against
      the goldens.

Exit criteria: every op implemented so far has a golden-output test that
passes against the dummy model.

## Stage 4 — RoPE And Attention

Scope: the hardest layer in the stack, built against the Stage 3 scaffolding.

- [ ] RoPE op: precomputed `inv_freq`, rotates pairs of dims on Q and K per
      head. Standalone, tested against golden.
- [ ] Q / K / V projections (three `LinearLayer`s; fused into one matmul
      later if useful).
- [ ] Scaled dot-product attention with causal mask, softmax over the last
      dim, attention-weighted V.
- [ ] Grouped-query attention from day one (`n_heads=2`, `n_kv_heads=1` in
      the toy): K and V repeated across query groups.
- [ ] Output projection.
- [ ] Tested layer-0 attention output matches golden within tolerance.

Exit criteria: a single attention layer applied to the toy model's input
matches the Python reference within `1e-4`.

## Stage 5 — Transformer Block And Full Model

Scope: compose everything into a model that produces logits, still on the
toy weights.

- [ ] Pre-norm transformer block: `x + attn(rmsnorm(x))`, `x + mlp(rmsnorm(x))`.
- [ ] `Model::forward(tokens) -> logits`: embedding → N blocks → final
      RMSNorm → LM head projection to vocab.
- [ ] Greedy argmax sampling.
- [ ] Generation loop: feed prompt tokens, sample next, append, repeat.

Exit criteria: full forward + greedy generation runs on the toy weights and
matches the Python reference logits at every layer boundary. Output text is
gibberish (random weights), but the machinery works end-to-end.

## Stage 6 — Real TinyLlama Weights

Scope: load actual TinyLlama from a HuggingFace checkpoint into the same
engine.

- [ ] Either: extend `load_weights` to parse `safetensors` directly (JSON
      header + raw fp32/fp16 blob), or add `tools/convert_safetensors.py`
      that re-packs HF weights into the custom format.
- [ ] Name mapping: HF parameter names → engine layers.
- [ ] Dtype handling: start fp32; convert from fp16 on load if needed.
- [ ] Sanity checks: norms of a few known weight matrices match a Python
      reference.

Exit criteria: model constructed from a real TinyLlama checkpoint; one
forward pass on a fixed input matches reference logits within small
tolerance.

## Stage 7 — Tokenizer

Scope: turn strings into token ids and back.

- [ ] Choose path:
  - Minimal: load TinyLlama's `tokenizer.model` (SentencePiece BPE) via a
    vendored small SentencePiece reader, or
  - Simpler: pre-encode prompts in Python and feed token ids to the engine
    while the C++ tokenizer matures.
- [ ] Implement encode / decode for the chat template used by TinyLlama-Chat
      (`<|system|>`, `<|user|>`, `<|assistant|>`).
- [ ] Round-trip test on a known string.

Exit criteria: `encode(decode(ids)) == ids` for a handful of prompts; chat
template produces the same ids as the HF tokenizer for a fixed example.

## Stage 8 — End-to-End Real Generation

Scope: glue real weights + real tokenizer + real generation.

- [ ] Greedy continuation of a real prompt with real TinyLlama weights.
- [ ] Match a Python reference for the first ~20 tokens (allow small
      numerical drift later in the sequence).

Exit criteria: a short prompt produces a coherent continuation that matches
the reference within tolerance.

## Stage 9 — KV Cache And Autoregressive Decoding

Scope: make generation practical. Up to here everything has been recompute-
from-scratch per token.

- [ ] Per-layer KV cache: shape `[num_kv_heads, max_seq, head_dim]`.
- [ ] Prefill path: process the whole prompt, fill the cache.
- [ ] Decode path: feed one token, append one row to K / V, attend over the
      cache.
- [ ] Position counter shared by RoPE and the cache writer.

Exit criteria: prefill + decode produces the same tokens as Stage 8 but at
much higher tokens/second, and memory is bounded by `max_seq`.

## Stage 10 — Sampling Controls

Scope: useful generation knobs.

- [ ] Temperature.
- [ ] Top-k.
- [ ] Top-p (nucleus).
- [ ] Repetition penalty (optional).
- [ ] Deterministic seed.

Exit criteria: same seed + same params → identical output across runs.

## Stage 11 — CPU Performance (Optional, Last)

Scope: make it fast without hiding the math.

- [ ] Cache-friendly matmul (blocked / loop-reordered).
- [ ] OpenMP parallelism on the outer matmul loop.
- [ ] Fused RMSNorm + projection where it stays readable.
- [ ] Shared activation arena (replace per-layer local scratch allocations).
- [ ] Optional: int8 / Q4 weight quantization for the linear layers.
- [ ] Lightweight benchmark harness reporting tokens/sec for prefill and
      decode.

Exit criteria: measurable speedup on prefill and decode with unchanged
outputs (within tolerance) and code that still reads like a teaching
implementation.

## Cross-Cutting Conventions

These hold from Stage 2 onward.

- **Read-only invariant for views.** Tensors backed by `mmap` regions must
  never be written to. Convention: `const Tensor&` parameters can be views;
  `Tensor&` outputs must be owned (vector-backed).
- **Goldens before ops.** From Stage 4 onward, every new op gets its
  reference output added to `tools/make_golden_outputs.py` before the C++
  implementation is written.
- **Scratch buffers are local until they hurt.** Allocate inside `forward()`
  through Stage 8. Switch to per-layer `mutable` scratch or a shared arena
  in Stage 11.

## Out Of Scope

- GPU / Metal / CUDA backends.
- Training, fine-tuning, LoRA.
- Multi-model orchestration, server / HTTP layer.
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
