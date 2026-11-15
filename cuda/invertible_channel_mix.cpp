#include <torch/extension.h>
#include <vector>
using namespace at;

// Channel mix via a learnable square matrix W applied to last dim
// x: [B, T, C], W: [C, C] => y: [B, T, C]
Tensor invertible_channel_mix_forward(const Tensor& x, const Tensor& W){
    TORCH_CHECK(x.dim()==3, "invertible_channel_mix expected 3D tensor");
    TORCH_CHECK(W.dim()==2, "W must be 2D");
    // use bmm for batched matmul: reshape to (B*T, C)
    int64_t B = x.size(0);
    int64_t T = x.size(1);
    int64_t C = x.size(2);
    Tensor x2 = x.view({B*T, C});
    Tensor y2 = at::matmul(x2, W);
    Tensor y = y2.view({B, T, C}).contiguous();
    return y;
}

// If needed, Python autograd will compute gradients automatically because we use ATen ops.

TORCH_LIBRARY(invertible_channel_mix, m) {
    m.def("forward(Tensor x, Tensor W) -> Tensor");
}

TORCH_LIBRARY_IMPL(invertible_channel_mix, CUDA, m) {
    m.impl("forward", TORCH_FN(invertible_channel_mix_forward));
}