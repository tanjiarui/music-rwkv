#include <torch/extension.h>
#include <vector>
using namespace at;

Tensor invertible_downsample_forward(const Tensor& x, int64_t stride){
    // x: [B, T, C] -> y: [B, T/stride, C*stride]
    TORCH_CHECK(x.dim()==3, "invertible_downsample expected 3D tensor");
    int64_t B = x.size(0);
    int64_t T = x.size(1);
    int64_t C = x.size(2);
    TORCH_CHECK(T % stride == 0, "T must be divisible by stride");
    int64_t newT = T / stride;
    // reshape by grouping consecutive tokens
    Tensor y = x.view({B, newT, stride * C}).contiguous();
    return y;
}

Tensor invertible_downsample_backward(const Tensor& grad_y, int64_t stride){
    // grad_y: [B, T/stride, C*stride] -> grad_x: [B, T, C]
    TORCH_CHECK(grad_y.dim()==3, "invertible_downsample backward expected 3D tensor");
    int64_t B = grad_y.size(0);
    int64_t newT = grad_y.size(1);
    int64_t SC = grad_y.size(2);
    TORCH_CHECK(SC % stride == 0, "channel dim must be multiple of stride");
    int64_t C = SC / stride;
    Tensor grad_x = grad_y.view({B, newT * stride, C}).contiguous();
    return grad_x;
}

TORCH_LIBRARY(invertible_downsample, m) {
    m.def("forward(Tensor x, int stride) -> Tensor");
    m.def("backward(Tensor y, int stride) -> Tensor");
}

TORCH_LIBRARY_IMPL(invertible_downsample, CUDA, m) {
    m.impl("forward", TORCH_FN(invertible_downsample_forward));
    m.impl("backward", TORCH_FN(invertible_downsample_backward));
}