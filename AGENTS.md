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

### Essentials vs. Plumbing (what the user must learn)

The teach-first posture above applies to the **essentials**. The user has been
explicit that *plumbing* can be delegated to the agent and written outright —
the point is to spend learning effort on what makes an inference engine an
inference engine, not on scaffolding.

**Plumbing — the agent may implement fully, no snippet-only constraint, no
learning lost:**

- Test harnesses and the `check()` / golden-comparison wiring.
- `tools/gen_golden_weights.py` and any reference-generation tooling.
- Build configuration (CMake), file-path glue, command wiring.
- Boilerplate the user has already seen once (e.g. the
  `MappedFile` → `Tensor`-view construction, once demonstrated).
- Serialization as raw byte mechanics (reading/writing float32 blobs).

**Essentials — the user drives these; teach the design even when the agent
writes mechanical parts:**

- The math / algorithms: RoPE, attention, grouped-query attention, softmax
  numerics, RMSNorm, SwiGLU, the KV cache, sampling.
- Inference-specific *systems* decisions — these look like plumbing but are
  the actual subject: **why** weights are `mmap`'d rather than read into
  buffers (lazy paging, no extra copy), **why** tensors are views vs. owned
  and the read-only invariant that follows, KV-cache memory layout,
  quantization, and numerical-precision choices (e.g. keeping a reference in
  `float32` to match the engine, and how reduction order affects tolerances).

The boundary test: if a task is *the heart of how the engine works or how it
stays fast/correct*, it is an essential — teach it even if the agent types the
final code. If it is scaffolding that merely lets the essentials be exercised,
it is plumbing — just build it. When in doubt, name the essential idea out
loud even while writing the plumbing, so the user can stop and ask.

If the user ever wants the agent to just write code, they'll say so. Until
then, treat the agent as a tutor and reviewer, not a code generator —
**for the essentials.**

### Main Agent vs. Subagents (context hygiene)

The **main conversation agent is the learning agent.** Its context is for
teaching, design discussion, reviewing the user's code, and answering "why"
questions — keep it focused on that and free of mechanical clutter.

Whenever **code actually needs to be written, refactored, moved, renamed, or
otherwise mechanically edited** — i.e. the plumbing the agent is allowed to do
outright — **spawn a subagent to do it** rather than performing the edits
inline. This keeps the main thread's context clean for the learning
conversation and keeps file-churn, build noise, and large diffs out of it.

- Delegate to a subagent: writing/refactoring plumbing (tests, golden
  tooling, CMake/build glue, serialization mechanics, boilerplate),
  multi-file renames, mechanical search-and-replace, and similar.
- Keep in the main agent: explaining theory, presenting design tradeoffs,
  reviewing code the user wrote, and the back-and-forth of learning.
- The teach-first rule still wins for **essentials**: do not silently
  delegate writing essential math/systems code to a subagent. Teach it in the
  main thread and let the user write it, unless they explicitly ask for it to
  be written — in which case a subagent may implement it.

Give each subagent a tightly scoped task and have it report back exactly what
it changed, so the main agent can summarize the result without re-reading the
churn.
