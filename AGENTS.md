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

## Build And Run

From the repository root:

```sh
cmake -S . -B build
cmake --build build
./build/magi_engine
```

If you need a clean release-style build, use a separate build directory such as
`build-release/` rather than changing generated files directly.

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
