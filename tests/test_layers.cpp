#include "layers.h"
#include "tensor.h"
#include "test_utils.h"

#include <utility>

void run_layers_tests(TestState& s) {
    // --- LinearLayer: y = x @ W^T, with W stored (out_features, in_features) ---
    // x (1,2), W (3,2) -> y (1,3). Also confirms get_weight_out_features() picks
    // the OUT dim (3), not the in dim (2).
    {
        Tensor W({ 3, 2 });
        fill(W, { 1, 0, 0, 1, 1, 1 });  // rows: [1,0], [0,1], [1,1]
        LinearLayer lin(std::move(W));

        Tensor x({ 1, 2 });
        fill(x, { 1, 2 });
        Tensor y({ 1, 3 });
        lin.forward(x, y);
        check(s, "linear: (1,2) @ W(3,2)^T", y, { 1, 2, 3 }, { 1, 3 });
    }

    // --- MLP SwiGLU, hand-computed (T=1, H=2, I=3) ---
    // x = [1, 2]
    //   gate = x @ Wg^T = [1, 2, 3]
    //   up   = x @ Wu^T = [3, 1, 2]
    //   A    = SiLU(gate) (.) up = [2.1931757, 1.7615942, 5.7154448]
    //   out  = A @ Wd^T = [7.9086205, 7.4770389]
    // (I=3 != H=2 on purpose, so a wrong intermediate dim would fail the asserts.)
    {
        Tensor Wg({ 3, 2 });
        fill(Wg, { 1, 0, 0, 1, 1, 1 });
        Tensor Wu({ 3, 2 });
        fill(Wu, { 1, 1, 1, 0, 0, 1 });
        Tensor Wd({ 2, 3 });
        fill(Wd, { 1, 0, 1, 0, 1, 1 });

        MLP mlp(LinearLayer(std::move(Wg)), LinearLayer(std::move(Wu)), LinearLayer(std::move(Wd)));

        Tensor x({ 1, 2 });
        fill(x, { 1, 2 });
        Tensor out({ 1, 2 });  // caller pre-allocates (T, H); MLP doesn't allocate OUT
        mlp.forward(x, out);
        check(s, "mlp swiglu: (1,2) hand-computed", out, { 7.9086205f, 7.4770389f }, { 1, 2 });
    }

    // --- MLP is shape-preserving for T > 1: two identical rows -> identical out ---
    {
        Tensor Wg({ 3, 2 });
        fill(Wg, { 1, 0, 0, 1, 1, 1 });
        Tensor Wu({ 3, 2 });
        fill(Wu, { 1, 1, 1, 0, 0, 1 });
        Tensor Wd({ 2, 3 });
        fill(Wd, { 1, 0, 1, 0, 1, 1 });

        MLP mlp(LinearLayer(std::move(Wg)), LinearLayer(std::move(Wu)), LinearLayer(std::move(Wd)));

        Tensor x({ 2, 2 });
        fill(x, { 1, 2, 1, 2 });  // row 0 == row 1
        Tensor out({ 2, 2 });
        mlp.forward(x, out);
        check(s, "mlp swiglu: (2,2) shape-preserving", out, { 7.9086205f, 7.4770389f, 7.9086205f, 7.4770389f },
              { 2, 2 });
    }
}
