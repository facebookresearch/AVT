// modified from
// https://github.com/pytorch/pytorch/blob/master/caffe2/operators/channelwise_conv3d_op_cudnn.cu
#include <ATen/ATen.h>
#include <THC/THCAtomics.cuh>
#include <stdio.h>
#include <math.h>
#include <float.h>

using namespace at;

// helper for cuda kernel
#define CUDA_KERNEL_LOOP(i, n)                                 \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); \
       i += blockDim.x * gridDim.x)

#define CUDA_NUM_THREADS 1024

inline int GET_BLOCKS(const int N)
{
  int kMaxGridNum = 65535;
  return std::min(kMaxGridNum, (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS);
}

// cuda kernel to compute the depthwise convolution forward
template <typename T>
__global__ void DepthwiseConv3dGPUKernelNCHW(
    const T* input, const T* filter, T* output,
    const int in_depth, const int in_length, const int in_rows, const int in_cols,
    const int filter_length, const int filter_rows, const int filter_cols,
    const int stride_length, const int stride_rows, const int stride_cols,
    const int pad_length, const int pad_rows, const int pad_cols,
    const int out_depth, const int out_length, const int out_rows, const int out_cols,
    int num_outputs) {

  CUDA_KERNEL_LOOP(thread_id, num_outputs) {
    const int OW = thread_id % out_cols;
    const int OH = (thread_id / out_cols) % out_rows;
    const int OL = (thread_id / out_cols / out_rows) % out_length;
    const int OC = (thread_id / out_cols / out_rows / out_length) % out_depth;
    const int OB = thread_id / out_cols / out_rows / out_length / out_depth;
    const int in_d = OC;

    const int input_offset_temp =
        (OB * in_depth + OC) * (in_length * in_rows * in_cols);
    const int input_row_start = OH * stride_rows - pad_rows;
    const int input_col_start = OW * stride_cols - pad_cols;
    const int input_length_start = OL * stride_length - pad_length;
    const int input_row_end = input_row_start + filter_rows;
    const int input_col_end = input_col_start + filter_cols;
    const int input_length_end = input_length_start + filter_length;
    const T* filter_start =
        filter + in_d * filter_rows * filter_cols * filter_length;

    T sum = 0;
    if (input_row_start >= 0 && input_col_start >= 0 &&
        input_length_start >= 0 && input_row_end < in_rows &&
        input_col_end < in_cols && input_length_end < in_length) {
// Loop that doesn't need to check for boundary conditions.
#pragma unroll
      for (int f_l = 0; f_l < filter_length; ++f_l) {
        const int in_l = input_length_start + f_l;
#pragma unroll
        for (int f_r = 0; f_r < filter_rows; ++f_r) {
          const int in_r = input_row_start + f_r;
          const T* filter_offset = filter_start +
              filter_cols * filter_rows * f_l + filter_cols * f_r;
#pragma unroll
          for (int f_c = 0; f_c < filter_cols; ++f_c) {
            const int in_c = input_col_start + f_c;

            const int input_offset = (input_offset_temp) +
                (in_l * in_cols * in_rows) + (in_r * in_cols) + in_c;
            sum += __ldg(input + input_offset) * __ldg(filter_offset + f_c);
          }
        }
      }
    } else {
// Loop that needs to check for boundary conditions.
#pragma unroll
      for (int f_l = 0; f_l < filter_length; ++f_l) {
        const int in_l = input_length_start + f_l;
#pragma unroll
        for (int f_r = 0; f_r < filter_rows; ++f_r) {
          const int in_r = input_row_start + f_r;
          const T* filter_offset = filter_start +
              filter_cols * filter_rows * f_l + filter_cols * f_r;
#pragma unroll
          for (int f_c = 0; f_c < filter_cols; ++f_c) {
            const int in_c = input_col_start + f_c;
            if (in_r >= 0 && in_r < in_rows && in_c >= 0 && in_c < in_cols &&
                in_l >= 0 && in_l < in_length) {
              const int input_offset = (input_offset_temp) +
                  (in_l * in_cols * in_rows) + (in_r * in_cols) + in_c;
              sum += __ldg(input + input_offset) * __ldg(filter_offset + f_c);
            }
          }
        }
      }
    }
    output[thread_id] = sum;
  }
}

// cuda kernel to compute the depthwise convolution backprop w.r.t. filter
// you probably don't want to understand the logic here
template <typename T>
__global__ void DepthwiseConv3dBackpropFilterGPUKernelNCHW(
    const T* out_backprop, const T* input, T* filter_backprop,
    const int in_depth, const int in_length, const int in_rows, const int in_cols,
    const int filter_length, const int filter_rows, const int filter_cols,
    const int stride_length, const int stride_rows, const int stride_cols,
    const int pad_length, const int pad_rows, const int pad_cols,
    const int out_depth, const int out_length, const int out_rows, const int out_cols,
    int num_out_backprop) {

  CUDA_KERNEL_LOOP(thread_id, num_out_backprop) {
    // Compute the indexes of this thread in the output.
    const int OW = thread_id % out_cols;
    const int OH = (thread_id / out_cols) % out_rows;
    const int OL = (thread_id / out_cols / out_rows) % out_length;
    const int OC = (thread_id / out_cols / out_rows / out_length) % out_depth;
    const int OB = thread_id / out_cols / out_rows / out_length / out_depth;

    // Compute the input depth and the index of depth multiplier.
    const int in_d = OC;

    // Decide if all input is valid, if yes, we can skip the boundary checks
    // for each input.
    const int in_r_start = OH * stride_rows - pad_rows;
    const int in_c_start = OW * stride_cols - pad_cols;
    const int in_l_start = OL * stride_length - pad_length;
    const int in_r_end = in_r_start + filter_rows;
    const int in_c_end = in_c_start + filter_cols;
    const int in_l_end = in_l_start + filter_length;

    const int out_backprop_offset =
        (OB * out_depth * out_length * out_rows * out_cols) +
        (OC * out_length * out_rows * out_cols) + (OL * out_rows * out_cols) +
        (OH * out_cols) + (OW);

    const T out_bp = __ldg(out_backprop + out_backprop_offset);
    if (in_r_start >= 0 && in_c_start >= 0 && in_r_end < in_rows &&
        in_c_end < in_cols && in_l_start >= 0 && in_l_end < in_length) {
#pragma unroll
      for (int f_l = 0; f_l < filter_length; ++f_l) {
        const int in_l = in_l_start + f_l;
#pragma unroll
        for (int f_r = 0; f_r < filter_rows; ++f_r) {
          const int in_r = in_r_start + f_r;
          // Avoid repeated computation.
          const int input_offset_temp =
              (OB * in_depth * in_length * in_rows * in_cols) +
              (OC * in_length * in_rows * in_cols) +
              (in_l * in_rows * in_cols) + (in_r * in_cols);

#pragma unroll
          for (int f_c = 0; f_c < filter_cols; ++f_c) {
            const int in_c = in_c_start + f_c;
            const int input_offset = input_offset_temp + in_c;
            T partial_sum = __ldg(input + input_offset) * out_bp;
            T* addr = filter_backprop +
                (in_d * filter_rows * filter_cols * filter_length) +
                (f_l * filter_rows * filter_cols) + (f_c + filter_cols * f_r);
            atomicAdd(addr, partial_sum);
          }
        }
      }
    } else {
#pragma unroll
      for (int f_l = 0; f_l < filter_length; ++f_l) {
        const int in_l = in_l_start + f_l;
#pragma unroll
        for (int f_r = 0; f_r < filter_rows; ++f_r) {
          const int in_r = in_r_start + f_r;
          // Avoid repeated computation.
          const int input_offset_temp =
              (OB * in_depth * in_length * in_rows * in_cols) +
              (OC * in_length * in_rows * in_cols) +
              (in_l * in_rows * in_cols) + (in_r * in_cols);
#pragma unroll
          for (int f_c = 0; f_c < filter_cols; ++f_c) {
            const int in_c = in_c_start + f_c;

            if (in_r >= 0 && in_r < in_rows && in_c >= 0 && in_c < in_cols &&
                in_l >= 0 && in_l < in_length) {
              const int input_offset = input_offset_temp + in_c;
              T partial_sum = __ldg(input + input_offset) * out_bp;
              T* addr = filter_backprop +
                  (in_d * filter_rows * filter_cols * filter_length) +
                  (f_l * filter_rows * filter_cols) + (f_c + filter_cols * f_r);
              atomicAdd(addr, partial_sum);
            }
          }
        }
      }
    }
  }
}

// cuda kernel to compute the depthwise convolution backprop w.r.t. input
// you probably don't want to understand the logic here
template <typename T>
__global__ void DepthwiseConv3dBackpropInputGPUKernelNCHW(
    const T* out_backprop, const T* filter, T* in_backprop,
    const int in_depth, const int in_length, const int in_rows, const int in_cols,
    const int filter_length, const int filter_rows, const int filter_cols,
    const int stride_length, const int stride_rows, const int stride_cols,
    const int pad_length, const int pad_rows, const int pad_cols,
    const int out_depth, const int out_length, const int out_rows, const int out_cols,
    int num_in_backprop) {

  CUDA_KERNEL_LOOP(thread_id, num_in_backprop) {
    const int IW = thread_id % in_cols;
    const int IH = (thread_id / in_cols) % in_rows;
    const int IL = (thread_id / in_cols / in_rows) % in_length;
    const int IC = (thread_id / in_cols / in_rows / in_length) % in_depth;
    const int IB = thread_id / in_cols / in_rows / in_length / in_depth;

    T sum = 0;

    const int out_r_start =
        max(0, (IH - filter_rows + pad_rows + stride_rows) / stride_rows);
    const int out_r_end = min(out_rows - 1, (IH + pad_rows) / stride_rows);
    const int out_c_start =
        max(0, (IW - filter_cols + pad_cols + stride_cols) / stride_cols);
    const int out_c_end = min(out_cols - 1, (IW + pad_cols) / stride_cols);
    const int out_l_start = max(
        0,
        (IL - filter_length + pad_length + stride_length) / stride_length);
    const int out_l_end =
        min(out_length - 1, (IL + pad_length) / stride_length);

#pragma unroll
    for (int out_l = out_l_start; out_l <= out_l_end; ++out_l) {
      const int f_l = IL + pad_length - out_l * stride_length;
      for (int out_r = out_r_start; out_r <= out_r_end; ++out_r) {
        const int f_r = IH + pad_rows - out_r * stride_rows;
        for (int out_c = out_c_start; out_c <= out_c_end; ++out_c) {
          const int f_c = IW + pad_cols - out_c * stride_cols;
          const int filter_offset =
              IC * filter_rows * filter_cols * filter_length +
              f_l * filter_cols * filter_rows + f_r * filter_cols + f_c;
          const int out_backprop_offset =
              (IB * out_depth * out_length * out_rows * out_cols) +
              (IC * out_length * out_rows * out_cols) +
              (out_l * out_rows * out_cols) + (out_r * out_cols) + (out_c);

          sum += __ldg(out_backprop + out_backprop_offset) *
              __ldg(filter + filter_offset);
        }
      }
    }
    const int in_backprop_offset =
        (IB * in_rows * in_cols * in_length * in_depth) +
        (IC * in_rows * in_cols * in_length) + (IL * in_rows * in_cols) +
        (IH * in_cols) + (IW);
    in_backprop[in_backprop_offset] = sum;
  }
}

// define the launchers
// forward prop
void DepthwiseConv3dForwardLauncher(
    const at::Tensor input, const at::Tensor weight, at::Tensor output,
    const int stride_t, const int stride_h, const int stride_w,
    const int pad_t, const int pad_h, const int pad_w) {
  // get all dims
  const int in_depth = input.size(1);
  const int in_length = input.size(2);
  const int in_rows = input.size(3);
  const int in_cols = input.size(4);

  const int out_depth = output.size(1);
  const int out_length = output.size(2);
  const int out_rows = output.size(3);
  const int out_cols = output.size(4);

  const int kernel_t = weight.size(2);
  const int kernel_h = weight.size(3);
  const int kernel_w = weight.size(4);

  int num_outputs = input.size(0) * out_depth * out_length * out_rows * out_cols;

  // launch the kernel
  AT_DISPATCH_FLOATING_TYPES(
    input.scalar_type(), "depthwise_conv3d_forward_cuda", ([&] {
      const scalar_t *input_ = input.data<scalar_t>();
      const scalar_t *filter_ = weight.data<scalar_t>();
      scalar_t *output_ = output.data<scalar_t>();
      DepthwiseConv3dGPUKernelNCHW<<<GET_BLOCKS(num_outputs), CUDA_NUM_THREADS>>>(
        input_, filter_, output_,
        in_depth, in_length, in_rows, in_cols,
        kernel_t, kernel_h, kernel_w,
        stride_t, stride_h, stride_w,
        pad_t, pad_h, pad_w,
        out_depth, out_length, out_rows, out_cols,
        num_outputs);
    }));

  THCudaCheck(cudaGetLastError());
}


// backward w.r.t. input
void DepthwiseConv3dBackwardInputLauncher(
    const at::Tensor grad_output, const at::Tensor weight, at::Tensor grad_input,
    const int stride_t, const int stride_h, const int stride_w,
    const int pad_t, const int pad_h, const int pad_w) {
  // get all dims
  const int in_depth = grad_input.size(1);
  const int in_length = grad_input.size(2);
  const int in_rows = grad_input.size(3);
  const int in_cols = grad_input.size(4);

  const int out_depth = grad_output.size(1);
  const int out_length = grad_output.size(2);
  const int out_rows = grad_output.size(3);
  const int out_cols = grad_output.size(4);

  const int kernel_t = weight.size(2);
  const int kernel_h = weight.size(3);
  const int kernel_w = weight.size(4);

  int num_inputs = grad_input.size(0) * in_depth * in_length * in_rows * in_cols;

  // launch the kernel
  AT_DISPATCH_FLOATING_TYPES(
    grad_output.scalar_type(), "depthwise_conv3d_backward_input_cuda", ([&] {
      const scalar_t *out_backprop_ = grad_output.data<scalar_t>();
      const scalar_t *filter_ = weight.data<scalar_t>();
      scalar_t *in_backprop_ = grad_input.data<scalar_t>();
      DepthwiseConv3dBackpropInputGPUKernelNCHW<<<GET_BLOCKS(num_inputs), CUDA_NUM_THREADS>>>(
        out_backprop_, filter_, in_backprop_,
        in_depth, in_length, in_rows, in_cols,
        kernel_t, kernel_h, kernel_w,
        stride_t, stride_h, stride_w,
        pad_t, pad_h, pad_w,
        out_depth, out_length, out_rows, out_cols,
        num_inputs);
    }));

  THCudaCheck(cudaGetLastError());
}

// backward w.r.t. weight
void DepthwiseConv3dBackwardWeightLauncher(
    const at::Tensor grad_output, const at::Tensor input, at::Tensor grad_weight,
    const int stride_t, const int stride_h, const int stride_w,
    const int pad_t, const int pad_h, const int pad_w) {
  // get all dims
  const int in_depth = input.size(1);
  const int in_length = input.size(2);
  const int in_rows = input.size(3);
  const int in_cols = input.size(4);

  const int out_depth = grad_output.size(1);
  const int out_length = grad_output.size(2);
  const int out_rows = grad_output.size(3);
  const int out_cols = grad_output.size(4);

  const int kernel_t = grad_weight.size(2);
  const int kernel_h = grad_weight.size(3);
  const int kernel_w = grad_weight.size(4);

  int num_outputs = input.size(0) * out_depth * out_length * out_rows * out_cols;

  // launch the kernel
  AT_DISPATCH_FLOATING_TYPES(
    input.scalar_type(), "depthwise_conv3d_forward_cuda", ([&] {
      const scalar_t *out_backprop_ = grad_output.data<scalar_t>();
      const scalar_t *input_ = input.data<scalar_t>();
      scalar_t *filter_backprop_ = grad_weight.data<scalar_t>();
      DepthwiseConv3dBackpropFilterGPUKernelNCHW<<<GET_BLOCKS(num_outputs), CUDA_NUM_THREADS>>>(
        out_backprop_, input_, filter_backprop_,
        in_depth, in_length, in_rows, in_cols,
        kernel_t, kernel_h, kernel_w,
        stride_t, stride_h, stride_w,
        pad_t, pad_h, pad_w,
        out_depth, out_length, out_rows, out_cols,
        num_outputs);
    }));

  THCudaCheck(cudaGetLastError());
}
