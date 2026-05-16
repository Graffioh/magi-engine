# Gated MLP

TinyLlama-style transformer blocks use a gated feed-forward network, often
called SwiGLU.

For hidden states:

$$
X \in \mathbb{R}^{T \times H}
$$

the MLP uses three projections:

$$
G = XW_g^\top
\qquad
G \in \mathbb{R}^{T \times I}
$$

$$
U = XW_u^\top
\qquad
U \in \mathbb{R}^{T \times I}
$$

$$
D = ZW_d^\top
\qquad
D \in \mathbb{R}^{T \times H}
$$

where:

- $H$ is hidden size.
- $I$ is intermediate size.
- $W_g$ is the gate projection.
- $W_u$ is the up projection.
- $W_d$ is the down projection.

## Gate

The gate branch is passed through SiLU:

$$
\operatorname{SiLU}(G)
=
G \odot \sigma(G)
$$

where $\odot$ is elementwise multiplication.

## Elementwise Product

The activated gate controls the up projection:

$$
Z =
\operatorname{SiLU}(G) \odot U
$$

Then the down projection returns to hidden size:

$$
\operatorname{GatedMLP}(X)
=
ZW_d^\top
$$

Expanded:

$$
\operatorname{GatedMLP}(X)
=
\left(
\operatorname{SiLU}(XW_g^\top)
\odot
XW_u^\top
\right)
W_d^\top
$$

The gate lets the model choose which intermediate features pass through, instead
of only applying a plain activation to a single projection.

Implemented in `GatedMLP::forward` in `src/layers.cpp`. The `*_proj.forward`
calls are the linear projection operation described in [Core Ops](ops.md); this
snippet shows the MLP-specific order of operations:

```cpp
void GatedMLP::forward(const Tensor& IN, Tensor& OUT) const {
  Tensor gate_out; // (seq_len, intermediate_size)
  gate_proj.forward(IN, gate_out);

  Tensor up_out; // (seq_len, intermediate_size)
  up_proj.forward(IN, up_out);

  SiLU(gate_out);

  Tensor hidden; // (seq_len, intermediate_size)
  mul(gate_out, up_out, hidden);

  down_proj.forward(hidden, OUT);
}
```
