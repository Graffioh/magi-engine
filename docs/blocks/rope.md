# RoPE

Rotary position embeddings encode token position by rotating pairs of query and
key dimensions. RoPE is applied to $Q$ and $K$, not to $V$.

For a head vector:

$$
x \in \mathbb{R}^{d_h}
$$

RoPE rotates pairs of dimensions:

$$
(x_{2m}, x_{2m+1})
$$

for:

$$
m = 0, 1, \dots, \frac{d_h}{2} - 1
$$

## Frequency

For pair index $m$, the inverse frequency is:

$$
\omega_m =
\theta^{-\frac{2m}{d_h}}
$$

Equivalently, when the implementation loops over even dimension index $i = 2m$:

$$
\omega_i =
\theta^{-\frac{i}{d_h}}
$$

The common base is:

$$
\theta = 10000
$$

Implementation in `SelfAttention::apply_rope`:

```cpp
float inv_freq = 1.0f / std::pow(
  theta,
  static_cast<float>(i) / static_cast<float>(head_dim)
);
```

## Rotation

At absolute position $p$, the angle is:

$$
\phi_{p,m} = p \omega_m
$$

The rotated pair is:

$$
x'_{2m}
=
x_{2m}\cos(\phi_{p,m})
-
x_{2m+1}\sin(\phi_{p,m})
$$

$$
x'_{2m+1}
=
x_{2m}\sin(\phi_{p,m})
+
x_{2m+1}\cos(\phi_{p,m})
$$

Implementation:

```cpp
float angle = static_cast<float>(absolute_pos) * inv_freq;
float c = std::cos(angle);
float s = std::sin(angle);

float x0 = x(pos, idx0);
float x1 = x(pos, idx1);

x(pos, idx0) = x0 * c - x1 * s;
x(pos, idx1) = x0 * s + x1 * c;
```

## Why It Matters

After RoPE, attention scores depend on relative positions because both query
and key vectors have been rotated according to their absolute token positions.

For autoregressive decoding, the position must include the cache offset:

$$
p = \operatorname{start\_pos} + t
$$

where $t$ is the position inside the current input chunk.

Implementation:

```cpp
for (int pos = 0; pos < seq_len; ++pos) {
  int absolute_pos = start_pos + pos;
}
```

## Layout Note

This project currently uses adjacent-pair rotation:

$$
(0,1), (2,3), (4,5), \dots
$$

Some LLaMA-compatible weight layouts use a split-half representation instead.
When real TinyLlama weights are loaded, the RoPE layout must match the weight
format.

Current adjacent-pair indexing:

```cpp
for (int i = 0; i < head_dim; i += 2) {
  int idx0 = head_offset + i;
  int idx1 = head_offset + i + 1;
}
```
