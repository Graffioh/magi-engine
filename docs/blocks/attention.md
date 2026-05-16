# Attention

Let the input hidden states be:

$$
X \in \mathbb{R}^{T \times H}
$$

where:

- $T$ is the sequence length.
- $H$ is the hidden size.
- $n_h$ is the number of query heads.
- $n_{kv}$ is the number of key/value heads.
- $d_h$ is the head dimension.
- $H = n_h d_h$.

## Projections

Self-attention first projects the hidden states into query, key, and value
tensors:

$$
Q = XW_q^\top
\qquad
Q \in \mathbb{R}^{T \times (n_h d_h)}
$$

$$
K = XW_k^\top
\qquad
K \in \mathbb{R}^{T \times (n_{kv} d_h)}
$$

$$
V = XW_v^\top
\qquad
V \in \mathbb{R}^{T \times (n_{kv} d_h)}
$$

Then view them as heads:

$$
Q \in \mathbb{R}^{T \times n_h \times d_h}
$$

$$
K, V \in \mathbb{R}^{T \times n_{kv} \times d_h}
$$

In `SelfAttention`, each projection is a `LinearLayer`. The projection
mechanism is implemented by `LinearLayer::forward` in `src/layers.cpp`:

```cpp
void LinearLayer::forward(const Tensor &IN, Tensor &OUT) const {
  // IN: (seq_len, input_dim)
  // W:  (output_dim, input_dim)
  // OUT: (seq_len, output_dim)
  matmul_nt(IN, W, OUT);

  if (has_bias()) {
    for (int i = 0; i < OUT.rows; ++i) {
      for (int j = 0; j < OUT.cols; ++j) {
        OUT(i, j) += bias[j];
      }
    }
  }
}
```

So each projected element is:

$$
OUT_{t,o}
=
\sum_{i=0}^{D_{in}-1}
IN_{t,i}W_{o,i}
$$

The Q/K/V projection call sites are:

```cpp
Tensor q, k, v;

q_proj.forward(IN, q); // q = IN * Wq^T, shape: (seq_len, num_heads * head_dim)
k_proj.forward(IN, k); // k = IN * Wk^T, shape: (seq_len, num_kv_heads * head_dim)
v_proj.forward(IN, v); // v = IN * Wv^T, shape: (seq_len, num_kv_heads * head_dim)
```

## Grouped-Query Attention

TinyLlama-style attention can use fewer key/value heads than query heads.
Several query heads share one key/value head.

The number of query heads per KV head is:

$$
g = \frac{n_h}{n_{kv}}
$$

For query head $h$, the shared KV head is:

$$
\operatorname{kv}(h) =
\left\lfloor \frac{h}{g} \right\rfloor
$$

Inside `SelfAttention::attn`, the grouped-query mapping is applied while
looping over query heads:

```cpp
const int kv_group_size = num_heads / num_kv_heads;

for (int q_head = 0; q_head < num_heads; ++q_head) {
  const int kv_head = q_head / kv_group_size;
  const int q_offset = q_head * head_dim;
  const int kv_offset = kv_head * head_dim;
}
```

## Attention Scores

For query head $h$, query position $i$, and key position $j$:

$$
s_{hij}
=
\frac{
Q_{i,h,:} \cdot K_{j,\operatorname{kv}(h),:}
}{
\sqrt{d_h}
}
$$

This is the scaled dot product between one query vector and one key vector.

Inside that same query-head loop, the score matrix for one head is filled by
the scaled dot product:

```cpp
for (int query_pos = 0; query_pos < seq_len; ++query_pos) {
  for (int key_pos = 0; key_pos < seq_len; ++key_pos) {
    float dot = 0.0f;
    for (int d = 0; d < head_dim; ++d) {
      dot += q(query_pos, q_offset + d) * k(key_pos, kv_offset + d);
    }
    scores(query_pos, key_pos) = dot * scale_factor;
  }
}
```

## Causal Mask

A token can only attend to itself and earlier tokens:

$$
\tilde{s}_{hij}
=
\begin{cases}
s_{hij}, & j \le i \\
-\infty, & j > i
\end{cases}
$$

The causal mask is implemented directly on the score matrix before softmax:

```cpp
void SelfAttention::causal_mask(Tensor& attn_scores) const {
  for (int i = 0; i < attn_scores.rows; ++i) {
    for (int j = i + 1; j < attn_scores.cols; ++j) {
      attn_scores(i, j) = -std::numeric_limits<float>::infinity();
    }
  }
}
```

## Softmax

The attention weights are computed with softmax over key positions:

$$
a_{hij}
=
\frac{
\exp(\tilde{s}_{hij})
}{
\sum\limits_{\ell=0}^{i}
\exp(\tilde{s}_{hi\ell})
}
$$

## Head Output

Each query head produces one output vector per token:

$$
O_{i,h,d}
=
\sum_{j=0}^{i}
a_{hij}
V_{j,\operatorname{kv}(h),d}
$$

After softmax, the same head writes its output into the matching slice of
`OUT`, which is the flattened concatenation of all query heads:

```cpp
for (int query_pos = 0; query_pos < seq_len; ++query_pos) {
  for (int d = 0; d < head_dim; ++d) {
    float weighted_sum = 0.0f;
    for (int key_pos = 0; key_pos < seq_len; ++key_pos) {
      weighted_sum += scores(query_pos, key_pos) * v(key_pos, kv_offset + d);
    }
    OUT(query_pos, q_offset + d) = weighted_sum;
  }
}
```

Equivalently, for one whole head:

$$
O_h
=
\operatorname{softmax}
\left(
\frac{Q_h K_{\operatorname{kv}(h)}^\top + M}{\sqrt{d_h}}
\right)
V_{\operatorname{kv}(h)}
$$

where:

$$
M_{ij}
=
\begin{cases}
0, & j \le i \\
-\infty, & j > i
\end{cases}
$$

## Concatenate Heads

The per-head outputs are concatenated back into hidden-size vectors:

$$
O_i
=
\operatorname{Concat}
\left(
O_{i,0,:},
O_{i,1,:},
\dots,
O_{i,n_h-1,:}
\right)
$$

So:

$$
O \in \mathbb{R}^{T \times H}
$$

## Output Projection

Finally, the concatenated attention output is projected back into the model
hidden space:

$$
Y = OW_o^\top
$$

This uses the same `LinearLayer::forward` projection described earlier, but with
the concatenated attention output as input:

$$
Y_{t,o}
=
\sum_{i=0}^{H-1}
O_{t,i}(W_o)_{o,i}
$$

The call site in `SelfAttention::forward` is:

```cpp
Tensor attn_out;
attn(q, k, v, attn_out);

o_proj.forward(attn_out, OUT);
```

## Compact Formula

The full self-attention layer is:

$$
\operatorname{SelfAttention}(X)
=
\operatorname{Concat}_{h=0}^{n_h-1}
\left[
\operatorname{softmax}
\left(
\frac{
Q_h K_{\operatorname{kv}(h)}^\top + M
}{
\sqrt{d_h}
}
\right)
V_{\operatorname{kv}(h)}
\right]
W_o^\top
$$

The important implementation detail is that attention is computed per query
head. The key/value head is selected by grouped-query attention, then each head
output is written into its slice of the final hidden-size tensor.
