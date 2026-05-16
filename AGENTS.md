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
