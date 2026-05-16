# AGENTS.md

Guidance for coding agents working in this repository.

## Project Purpose

This is a small C++ playground for learning how an inference engine works from
first principles. The target model family is TinyLlama-style transformer
inference, CPU only.

Favor clarity over cleverness. The code should make the mechanics of inference
visible: tensors, matrix operations, embeddings, attention, normalization,
MLPs, tokenization, weight loading, and the autoregressive generation loop.

## Core Constraints

- Keep the implementation CPU-only.
- Use C++20 and the existing CMake project.
- Avoid adding large frameworks or GPU/runtime dependencies.
- Prefer simple standard-library data structures unless a local abstraction
  clearly improves understanding.
- Preserve the educational nature of the code. Optimizations are welcome when
  they are explained by the surrounding code structure, not hidden behind magic.

## Repository Layout

- `CMakeLists.txt`: build configuration for the `magi_engine` executable.
- `README.md`: user-facing build and run notes.
- `docs/`: learning notes that explain inference blocks with formulas.
- `docs/blocks/`: one Markdown note per important implementation concept.
- `src/tensor.h`: basic tensor storage and indexing.
- `src/ops.*`: low-level tensor/math operations such as matmul, softmax, RMSNorm.
- `src/layers.*`: transformer building blocks.
- `src/model.*`: model-level orchestration.
- `src/tokenizer.*`: tokenization support.
- `src/weights.*`: model weight loading/support.
- `build/` and `build-release/`: generated build directories; do not edit by hand.

## Build And Run

From the repository root:

```sh
cmake -S . -B build
cmake --build build
./build/magi_engine
```

If you need a clean release-style build, use a separate build directory such as
`build-release/` rather than changing generated files directly.

## Coding Style

- Match the style already present in nearby files.
- Keep tensor shape comments near transformer math, especially for projections,
  attention scores, KV heads, and MLP intermediate tensors.
- Validate dimensions before math operations and throw standard exceptions with
  useful messages.
- Use descriptive names for model concepts: `hidden_size`, `num_heads`,
  `head_dim`, `seq_len`, `vocab_size`, `intermediate_size`.
- Keep comments focused on inference concepts or non-obvious math.
- Avoid unrelated refactors while implementing a learning step.

## Explanations And References

When explaining AI theory topics, include paper references or similar primary
sources when possible. Prefer concise citations that help the reader connect the
implementation to the underlying idea, such as the original Transformer paper,
RoPE, RMSNorm, LLaMA/TinyLlama, or grouped-query attention papers.

## Learning Notes

- Keep `docs/blocks/` in sync with the code as the engine grows.
- When implementing a new inference concept, add or update the corresponding
  Markdown note with the key shapes, formulas, and a short explanation.
- Include small C++ snippets from the real implementation so readers can connect
  the formulas back to the code.
- Keep each snippet local to the subsection's concept. For example, a
  projection section should show the projection helper and its Q/K/V call sites,
  not the entire attention forward pass.
- If a snippet calls a helper that hides the relevant math, also show the helper
  implementation or replace the snippet with the lower-level code.
- Avoid isolated constructor details unless initialization or shape setup is the
  concept being explained.
- When changing behavior in an existing block, update its note in the same
  change so the educational docs do not drift from the implementation.
- Prefer one focused note per concept, such as attention, RoPE, Gated MLP,
  tensor storage, normalization, tokenization, KV cache, or sampling.

## Inference Engine Priorities

When extending the engine, prefer this order of progress:

1. Correctness and readable tensor shapes.
2. A complete minimal TinyLlama-style forward pass.
3. Tokenizer and weight-loading compatibility.
4. Deterministic generation behavior.
5. CPU performance improvements that keep the code understandable.

Useful implementation milestones include:

- RMSNorm, SiLU, softmax, matmul variants.
- Embedding lookup and final language-model head.
- RoPE for query/key projections.
- Causal self-attention with grouped-query attention support.
- KV cache for autoregressive decoding.
- Transformer block composition.
- Token sampling controls such as temperature and top-k/top-p.

## Testing And Verification

There is not yet a formal test suite. When changing behavior:

- Build with `cmake --build build`.
- Run `./build/magi_engine`.
- Add small deterministic checks when introducing math-heavy code.
- Compare simple tensor outputs against hand-computed expectations where useful.
- Keep benchmark/demo code separate from core operations when possible.

## Dependency Policy

Before adding a dependency, ask whether it helps teach the inference engine. A
small, well-contained helper dependency may be acceptable for file formats or
tokenization, but core transformer math should remain visible in this codebase.

## Agent Etiquette

- Do not modify generated build artifacts unless explicitly asked.
- Do not remove user experiments without permission.
- Keep changes scoped to the requested learning or implementation step.
- If a design choice is ambiguous, choose the simpler educational path and note
  the tradeoff.
