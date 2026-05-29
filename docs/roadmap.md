# Roadmap

Goal: a CPU-only C++ inference engine that runs TinyLlama
(`TinyLlama-1.1B-Chat-v1.0`) end-to-end, from a tokenized prompt to streamed
output tokens. The roadmap is staged so each milestone produces something
runnable and verifiable on its own.

Target model reference: TinyLlama 1.1B (22 layers, hidden 2048, 32 query heads,
4 KV heads, head_dim 64, intermediate 5632, vocab 32000, RoPE base 10000,
RMSNorm, SwiGLU MLP, grouped-query attention).

The big structural idea: **golden reference outputs come before attention;
real weight-file loading comes last, just before real inference.** Attention
(RoPE + grouped-query + causal mask + softmax) is the easiest piece to get
subtly wrong while still producing finite, plausible-looking numbers, and
hand-computing its expected output is brutal — so a Python-computed golden
reference comes first, and every layer is diffed against it. But that
reference does **not** require a real weight loader: the toy model runs on
weights generated in Python and fed through the simplest possible
serialization, so the whole math stack (embedding → attention → MLP → logits
→ greedy sampling) is built and verified end-to-end before any model-file
parsing exists. Weight loading is I/O plumbing, not math — it is deferred and
becomes the bridge to *real* inference.

> **Plan revision (2026-05).** Two changes from the original plan: (1) the
> model file format is **GGUF** (llama.cpp's format), F32 tensors only to
> start (zero-copy mmap views), rather than a homemade format or safetensors —
> real TinyLlama GGUFs exist to download, and a GGUF carries the model
> hyperparameters *and* the tokenizer inside it. (2) The build order is
> reordered so **attention and the full toy model are built before the GGUF
> loader**, against Python goldens. The mmap infrastructure (`Tensor` storage
> refactor + `MappedFile`) is already built and tested — it simply isn't
> *used* until the GGUF loader lands later.
>
> Revised order from here: **golden harness (simple serialization)** →
> **RoPE + attention** → **transformer block + full toy model + greedy
> generation** (all vs goldens) → **GGUF loader (F32)** → **real TinyLlama
> weights** → **tokenizer (from GGUF metadata)** → **end-to-end real
> generation**.

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
- [x] Token embedding lookup (`out = embed_weights[token_id]`), via
      `ops::embed` + thin `EmbeddingLayer` wrapper.
- [x] Hand-computed tests for `MLP` in `tests/`.
- [x] Hand-computed test for embedding in `tests/`.

Exit criteria: scratchpad constructs an `MLP` from random tensors, runs
`forward(x)` on a random input, prints a finite, correctly-shaped output.

## Stage 2 — Storage And Mmap Infrastructure (done)

Scope: backing storage that can either own its data or view into an mmap'd
region. Built early because both pieces were small and unblocked. The file
*parsing* that consumes this (`load_weights`) is deferred to the GGUF stage
below — these are the foundations it will stand on.

- [x] `Tensor` refactor: backing storage becomes
      `float* + std::shared_ptr<void>` so a tensor can either own
      its data (vector-backed) or view into an mmap'd region. Existing tests
      keep passing unchanged.
- [x] `MappedFile` RAII wrapper (`open` → `fstat` → `mmap(PROT_READ)` →
      `munmap`/`close` in destructor). POSIX-only. (Has its own smoke test in
      `tests/test_mmap.cpp`; named `MappedFile` rather than `MmapFile`.)

Exit criteria (met): a `MappedFile` maps a raw float blob and a `Tensor` view
over it reads back the expected values (`tests/test_mmap.cpp`).

## Stage 3 — Golden Reference Harness

Scope: Python-computed reference outputs for every op, decoupled from any real
file format. This is the ground truth that makes attention debuggable. The
serialization is deliberately the dumbest thing that works — **not** GGUF —
so it can exist before the loader does.

- [x] Toy dims: `vocab=32`, `hidden=8`, `intermediate=16`, `n_layers=2`,
      `n_heads=2`, `n_kv_heads=1`, `head_dim=4`, `max_seq=16`. Small enough
      to print whole tensors, structured enough to exercise GQA and stacking.
- [x] `tools/gen_golden_weights.py`: NumPy-based, fixed seed. Generates the toy weight
      set *and* a reference forward pass; emits both the weights and the
      expected outputs (embedding, RMSNorm, MLP, and later RoPE / attention).
- [x] Serialization: raw little-endian `float32` dumps (one blob per tensor,
      shapes known by the test) read back through the existing `MappedFile`,
      or a generated C++ header of `static const float[]`. No magic, no
      directory — that complexity is GGUF's job later.
- [x] Tests load the toy weights + goldens and check `MLP` / embedding (and
      later attention / full-model) outputs within tolerance.

Exit criteria: every op implemented so far has a golden-output test that
passes against the toy reference.

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

## Stage 6 — GGUF Loading And Real TinyLlama Weights

Scope: the deferred I/O plumbing, then real weights through it. This is the
bridge from toy-model-in-memory to a real checkpoint. Build the loader against
a tiny F32 GGUF first (written by Python `gguf`), then point it at the real
model.

GGUF loader (F32 tensors only to start):

- [ ] Header parse: magic `"GGUF"`, `version`, `tensor_count`,
      `metadata_kv_count` (all little-endian).
- [ ] Metadata KV parser: typed values incl. arrays (`read_metadata_value`
      dispatch on the type enum). Extract hyperparameters — `block_count`,
      head counts, `embedding_length`, RoPE `freq_base`, RMSNorm eps.
- [ ] Tensor directory parse: name, dims, `ggml_type` (assert F32 == 0),
      `offset`. Reverse ggml dim order (`ne[0]` is the inner/fast dim).
- [ ] Build `Tensor` mmap views: data section start padded to
      `general.alignment` (default 32); each view at `base + data_start +
      offset`, sharing the `MappedFile` via `owner_`. Returns
      `std::unordered_map<std::string, Tensor>`.
- [ ] Round-trip test: a tiny F32 GGUF written by Python `gguf`, loaded in
      C++, values match the NumPy source.

Real weights:

- [ ] Obtain an F32 TinyLlama GGUF (convert from HF with llama.cpp's
      `convert_hf_to_gguf.py --outtype f32`, or download one).
- [ ] Name mapping: ggml names (`blk.{i}.attn_q.weight`, `token_embd.weight`,
      `output_norm.weight`, …) → engine layers.
- [ ] Sanity checks: norms of a few known weight matrices match a Python
      reference.

Exit criteria: model constructed from a real TinyLlama GGUF; one forward pass
on a fixed input matches reference logits within small tolerance.

## Stage 7 — Tokenizer

Scope: turn strings into token ids and back.

- [ ] Choose path:
  - Preferred: pull the vocab / scores / merges straight from the **GGUF
    metadata** parsed in Stage 6 (`tokenizer.ggml.*` keys) — no separate
    tokenizer file needed, and
  - Simpler stopgap: pre-encode prompts in Python and feed token ids to the
    engine while the C++ tokenizer matures.
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
