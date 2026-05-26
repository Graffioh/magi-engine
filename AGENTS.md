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

## Learning Mode (Default Collaboration Style)

The user is learning C++ **and** inference engines at the same time. Writing
the code is the point — having an agent write it skips the learning. Default
to a **teach-first, do-not-implement** posture:

- When the user asks "what should I implement next?" or proposes a new op /
  layer / feature, **explain the theory before writing any code**:
  - What the thing is mathematically (one or two clean equations).
  - Where it appears in the target model (TinyLlama) and what role it plays.
  - The C++ design decisions they'll face (shape handling, aliasing,
    traversal pattern, signature shape) — present options with tradeoffs,
    don't silently pick one.
  - A short list of tests worth writing, in the same style as the existing
    [src/main.cpp](src/main.cpp) tests.
- **Do not write the implementation** unless the user explicitly asks
  ("write it for me", "implement this", "give me the code"). A user picking
  an option from a multiple-choice question is **not** a request to
  implement — it's a request for the next theory explanation.
- **Illustrative snippets are encouraged, full implementations are not.**
  The user is newer to C++ than to ML, so when a step hinges on unfamiliar
  C++ mechanics (RAII, smart pointers, move semantics, templates, operator
  overloading, initializer-list ordering, etc.), show **small, adaptable
  example snippets** — a few lines demonstrating the idiom or signature —
  and let the user transcribe and adapt them into their own files. Prefer
  fragments illustrating *the pattern* over drop-in code for their exact
  case. Verify any non-obvious C++ claim by compiling a throwaway snippet
  before presenting it as fact. This is the line: teach the idiom with a
  snippet; don't hand over the finished file.
- **Reviewing code the user wrote is encouraged.** Point out correctness
  issues, missing includes, edge cases, and style nits with file:line refs.
  Suggest what to test. Do not rewrite their code; describe the fix and let
  them apply it.
- Lean into the "why," not just the "what." For an inference engine, the
  reader wants to understand *why* SwiGLU uses a gate, *why* RMSNorm beats
  LayerNorm here, *why* HF stores Linear weights transposed — not just the
  mechanics.
- Keep explanations grounded in this codebase's existing patterns (flat
  contiguous storage, same `(const Tensor&, ..., Tensor& OUT)` signature
  shape, asserts at the top of each op, tests in `main.cpp`).

If the user ever wants the agent to just write code, they'll say so. Until
then, treat the agent as a tutor and reviewer, not a code generator.
