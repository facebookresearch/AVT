#include <torch/extension.h>

#include <cmath>
#include <vector>

void DepthwiseConv3dForwardLauncher(
  const at::Tensor input, const at::Tensor weight, at::Tensor output,
  const int stride_t, const int stride_h, const int stride_w,
  const int pad_t, const int pad_h, const int pad_w);

void DepthwiseConv3dBackwardInputLauncher(
  const at::Tensor grad_output, const at::Tensor weight, at::Tensor grad_input,
  const int stride_t, const int stride_h, const int stride_w,
  const int pad_t, const int pad_h, const int pad_w);

void DepthwiseConv3dBackwardWeightLauncher(
  const at::Tensor grad_output, const at::Tensor input, at::Tensor grad_weight,
  const int stride_t, const int stride_h, const int stride_w,
  const int pad_t, const int pad_h, const int pad_w);

#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x, " must be a CUDAtensor ")
#define CHECK_CONTIGUOUS(x) \
  TORCH_CHECK(x.is_contiguous(), #x, " must be contiguous ")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)

void depthwise_conv3d_forward_cuda(
    at::Tensor input, at::Tensor weight, at::Tensor output,
    const int stride_t, const int stride_h, const int stride_w,
    const int pad_t, const int pad_h, const int pad_w) {
  // sanity check
  CHECK_INPUT(input);
  CHECK_INPUT(weight);
  // call the launcher
  DepthwiseConv3dForwardLauncher(
    input, weight, output,
    stride_t, stride_h, stride_w,
    pad_t, pad_h, pad_w);
}

void depthwise_conv3d_backward_input_cuda(
    const at::Tensor grad_output, const at::Tensor weight, at::Tensor grad_input,
    const int stride_t, const int stride_h, const int stride_w,
    const int pad_t, const int pad_h, const int pad_w) {
  // sanity check
  CHECK_INPUT(grad_output);
  CHECK_INPUT(weight);

  // call the launcher
  DepthwiseConv3dBackwardInputLauncher(
    grad_output, weight, grad_input,
    stride_t, stride_h, stride_w,
    pad_t, pad_h, pad_w);
}

void depthwise_conv3d_backward_weight_cuda(
    const at::Tensor grad_output, const at::Tensor input, at::Tensor grad_weight,
    const int stride_t, const int stride_h, const int stride_w,
    const int pad_t, const int pad_h, const int pad_w) {
  // sanity check
  CHECK_INPUT(grad_output);
  CHECK_INPUT(input);
  // call the launcher
  DepthwiseConv3dBackwardWeightLauncher(
    grad_output, input, grad_weight,
    stride_t, stride_h, stride_w,
    pad_t, pad_h, pad_w);
}

// bind to python
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("depthwise_conv3d_forward_cuda", &depthwise_conv3d_forward_cuda,
        "Depthwise Conv3d Forward (CUDA)");
  m.def("depthwise_conv3d_backward_input_cuda", &depthwise_conv3d_backward_input_cuda,
        "Depthwise Conv3d Backward Input (CUDA)");
  m.def("depthwise_conv3d_backward_weight_cuda", &depthwise_conv3d_backward_weight_cuda,
        "Depthwise Conv3d Backward Weight (CUDA)");
}
