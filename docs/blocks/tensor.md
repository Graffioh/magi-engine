# Tensor

The engine currently uses a small 2D tensor abstraction:

$$
X \in \mathbb{R}^{R \times C}
$$

where:

- $R$ is the number of rows.
- $C$ is the number of columns.
- Values are stored as contiguous `float` data in row-major order.

## Storage

For row $r$ and column $c$, the flat storage index is:

$$
\operatorname{index}(r, c) = rC + c
$$

So:

$$
X_{r,c} = \operatorname{data}[rC + c]
$$

This makes each row contiguous in memory. For transformer inference, a common
2D convention in this project is:

$$
X \in \mathbb{R}^{T \times H}
$$

where:

- $T$ is sequence length.
- $H$ is hidden size.

## Flattened Heads

Multi-head tensors are often stored in 2D form, with heads flattened into the
column dimension:

$$
Q \in \mathbb{R}^{T \times (n_h d_h)}
$$

This can be viewed conceptually as:

$$
Q \in \mathbb{R}^{T \times n_h \times d_h}
$$

The 2D column for head $h$ and head dimension $d$ is:

$$
c = h d_h + d
$$

So:

$$
Q_{t,h,d} = Q_{t, h d_h + d}
$$

This lets the implementation stay simple while still exposing the transformer
shape logic.

## Implementation Reference

Defined in `src/tensor.h`:

```cpp
struct Tensor {
    std::vector<float> data;
    int rows;
    int cols;

    Tensor(int rows = 0, int cols = 0) : rows(rows), cols(cols) {
        data.resize(rows * cols, 0.0f);
    }
};
```

The indexing operators are the row-major formula from above:

```cpp
float& operator()(int r, int c) {
    return data[r * cols + c];
}

const float& operator()(int r, int c) const {
    return data[r * cols + c];
}
```

When code writes `tensor(row, col)`, it is really reading or writing:

$$
\operatorname{data}[row \cdot cols + col]
$$
