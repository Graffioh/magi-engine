# Core Ops

The low-level ops are intentionally small and visible. They are the building
blocks used by layers.

## Matrix Multiplication

For:

$$
A \in \mathbb{R}^{M \times K}
\qquad
B \in \mathbb{R}^{K \times N}
$$

the output is:

$$
C = AB
\qquad
C \in \mathbb{R}^{M \times N}
$$

Elementwise:

$$
C_{i,j} =
\sum_{k=0}^{K-1}
A_{i,k} B_{k,j}
$$

## Matrix Multiplication With Transposed Right Side

Linear layers store weights as:

$$
W \in \mathbb{R}^{D_{out} \times D_{in}}
$$

For input:

$$
X \in \mathbb{R}^{T \times D_{in}}
$$

the linear output is:

$$
Y = XW^\top
\qquad
Y \in \mathbb{R}^{T \times D_{out}}
$$

Elementwise:

$$
Y_{t,o} =
\sum_{i=0}^{D_{in}-1}
X_{t,i} W_{o,i}
$$

Implementation in `src/ops.cpp`:

```cpp
void matmul_nt(const Tensor& A, const Tensor& B, Tensor& OUT) {
    if (A.cols != B.cols) {
        throw std::invalid_argument("Incompatible dimensions for matmul_nt");
    }
    OUT = Tensor(A.rows, B.rows);

    for (int i = 0; i < OUT.rows; ++i) {
        for (int j = 0; j < OUT.cols; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < A.cols; ++k) {
                sum += A(i, k) * B(j, k);
            }
            OUT(i, j) = sum;
        }
    }
}
```

The layer-level projection wrapper is shown here because it is only a thin call
into `matmul_nt`; there is no extra hidden projection math:

```cpp
void LinearLayer::forward(const Tensor &IN, Tensor &OUT) const {
  matmul_nt(IN, W, OUT);
}
```

## Softmax

Softmax converts scores into normalized weights. For a row vector $x$:

$$
\operatorname{softmax}(x)_j =
\frac{\exp(x_j)}
{\sum_k \exp(x_k)}
$$

The implementation subtracts the row maximum for numerical stability:

$$
\operatorname{softmax}(x)_j =
\frac{\exp(x_j - m)}
{\sum_k \exp(x_k - m)}
\qquad
m = \max_k x_k
$$

Implementation in `src/ops.cpp`:

```cpp
void softmax(Tensor& IN) {
    for (int i = 0; i < IN.rows; ++i) {
        float max_val = -std::numeric_limits<float>::infinity();
        for (int j = 0; j < IN.cols; ++j) {
            if (IN(i, j) > max_val) {
                max_val = IN(i, j);
            }
        }

        float sum_exp = 0.0f;
        for (int j = 0; j < IN.cols; ++j) {
            sum_exp += std::exp(IN(i, j) - max_val);
        }

        for (int j = 0; j < IN.cols; ++j) {
            IN(i, j) = std::exp(IN(i, j) - max_val) / sum_exp;
        }
    }
}
```

## RMSNorm

RMSNorm normalizes each row by its root mean square, then scales by learned
weights.

For one row $x \in \mathbb{R}^{H}$:

$$
\operatorname{rms}(x) =
\sqrt{
\frac{1}{H}
\sum_{j=0}^{H-1} x_j^2
+ \epsilon
}
$$

Then:

$$
y_j =
\frac{x_j}{\operatorname{rms}(x)}
w_j
$$

RMSNorm is used in LLaMA-style transformer blocks instead of LayerNorm.

Implementation in `src/ops.cpp`:

```cpp
const float mean_squares = sum_squares / IN.cols;
const float inv_rms = 1.0f / std::sqrt(mean_squares + epsilon);

for (int j = 0; j < IN.cols; ++j) {
    IN(i, j) = IN(i, j) * inv_rms * weight[j];
}
```

## SiLU

SiLU is the activation used inside the gated MLP:

$$
\operatorname{SiLU}(x) =
x \sigma(x)
=
\frac{x}{1 + e^{-x}}
$$

Implementation in `src/ops.cpp`:

```cpp
void SiLU(Tensor& IN) {
    for (int i = 0; i < IN.rows * IN.cols; ++i) {
        float x = IN.data[i];
        IN.data[i] = x / (1.0f + std::exp(-x));
    }
}
```
