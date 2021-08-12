from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='depthwise_conv3d',
    ext_modules=[
        CUDAExtension('depthwise_conv3d_cuda',
            ['src/depthwise_conv3d_cuda.cpp',
            'src/depthwise_conv3d_cuda_kernel.cu'],
            extra_compile_args={
            'cxx': ['-Wno-unused-function', '-std=c++14'],
            'nvcc': ['-O2',
                '-std=c++14',
                '-D__CUDA_NO_HALF_OPERATORS__',
                '-D__CUDA_NO_HALF_CONVERSIONS__',
                '-D__CUDA_NO_HALF2_OPERATORS__',
                # you need to modify this for your GPU arch
                "-gencode=arch=compute_60,code=\"sm_60\"",  # For Quadro GP100
                "-gencode=arch=compute_70,code=\"sm_70\""]  # For Volta V100
            }
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
