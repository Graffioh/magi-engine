# Learning Notes

This directory explains the inference engine block by block. Each note should
connect the C++ implementation to the relevant tensor shapes and formulas.

The goal is not to be exhaustive. The goal is to make the next implementation
step easier to read, debug, and extend.

## Blocks

- [Tensor](blocks/tensor.md): storage, indexing, and shape conventions.
- [Core Ops](blocks/ops.md): matmul, softmax, RMSNorm, SiLU, and elementwise ops.
- [Attention](blocks/attention.md): causal self-attention with grouped-query attention.
- [RoPE](blocks/rope.md): rotary position embeddings for query/key heads.
- [Gated MLP](blocks/gated_mlp.md): SwiGLU-style feed-forward network.

## Maintenance Rule

When a new inference concept is implemented or an existing one changes, update
the matching note in `docs/blocks/`. If no note exists yet, add one.

## Note Style

Each section should pair the formula with the smallest useful C++ snippet for
that exact concept. Do not use a broad surrounding method when the section is
about one operation.

Good examples:

- A projection section should show `LinearLayer::forward` and the Q/K/V call
  sites.
- A RoPE frequency section should show the `inv_freq` calculation.
- A causal mask section should show only the mask loop.

If a code snippet calls a helper that hides the important math, show the helper
too or use a lower-level snippet. Avoid constructor snippets unless the section
is specifically about initialization or shape setup.
