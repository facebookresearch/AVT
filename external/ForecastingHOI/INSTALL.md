# Installation

```shell
git clone https://github.com/happyharrycn/video_analysis.git
```

## Requirements

- Linux
- Python 3.5+
- NumPy 1.11+
- OpenCV (Either 2/3/4 should work)
- PyTorch 1.2
- CUDA 9.0+
- NCCL 2+ (for distributed training)
- ffmpeg 4.0+
- Lintel 1.0+
- Tensorboard 1.14+ (for visualization)


If you are using Anaconda, try

```shell
conda install pytorch torchvision cudatoolkit=9.0 -c pytorch
conda install -c conda-forge ffmpeg opencv lintel tensorboardx
```

if you plan to use the CSN model from Facebook, you will need to compile the CUDA code for 3D depthwise convolution.

```shell
module load cuda/10.2  # Note it errors with cuda/10.1, some string errors, like https://github.com/pytorch/vision/issues/1893
cd ./libs/models/depthwise_conv3d
python setup.py build_ext --inplace
```

**Make sure that you modify the gencode in setup.py to match your GPU arch**
