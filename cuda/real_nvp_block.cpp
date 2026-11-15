#include <torch/extension.h>
#include <vector>

using namespace at;

std::vector<Tensor> real_nvp_forward_impl(const Tensor& x, const Tensor& s, const Tensor& t, bool reverse){
    TORCH_CHECK(x.dim() == 3, "real_nvp expects 3D x");
    int64_t C = x.size(2);
    TORCH_CHECK((C % 2) == 0, "channel dim must be even");
    int64_t C2 = C / 2;
    TORCH_CHECK(s.sizes() == IntArrayRef({x.size(0), x.size(1), C2}), "s shape mismatch");
    TORCH_CHECK(t.sizes() == IntArrayRef({x.size(0), x.size(1), C2}), "t shape mismatch");

    Tensor x1 = x.narrow(2, 0, C2);
    Tensor x2 = x.narrow(2, C2, C2);
    if (!reverse) {
        Tensor y2 = x2 * at::exp(s) + t;
        Tensor y = at::cat({x1, y2}, 2);
        Tensor logdet = s.sum({1,2}); // [B]
        return {y.contiguous(), logdet.contiguous()};
    } else {
        Tensor y2 = (x2 - t) * at::exp(-s);
        Tensor y = at::cat({x1, y2}, 2);
        Tensor logdet = (-s).sum({1,2});
        return {y.contiguous(), logdet.contiguous()};
    }
}

// Python-facing wrapper with checks (keeps same signature for pybind and TorchLibrary)
std::vector<Tensor> forward(torch::Tensor x, torch::Tensor s, torch::Tensor t, bool reverse) {
    // allow float32 or bfloat16 depending on your pipeline; adjust checks if needed:
    TORCH_CHECK(x.is_cuda() == torch::cuda::is_available(), "x device mismatch");
    TORCH_CHECK(s.is_cuda() == x.is_cuda(), "s device mismatch");
    TORCH_CHECK(t.is_cuda() == x.is_cuda(), "t device mismatch");
    return real_nvp_forward_impl(x, s, t, reverse);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "RealNVP forward (x, s, t, reverse)");
}

// Register as torch.ops namespace so torch.compile / dynamo can recognize it
TORCH_LIBRARY(real_nvp, m) {
    m.def("forward(Tensor x, Tensor s, Tensor t, bool reverse) -> (Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(real_nvp, CUDA, m) {
    m.impl("forward", &forward);
}