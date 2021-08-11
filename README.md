
# Anticipative Video Transformer

<p><img src="https://rohitgirdhar.github.io/DetectAndTrack/assets/cup.png" width="30px" align="center" /> Ranked <b>first</b> in the keypoint tracking task of the <a href="https://epic-kitchens.github.io/2021#results">CVPR 2021 EPIC-Kitchens Challenge</a>! (entry: AVT-FB-UT)</p>

[[project page](https://facebookresearch.github.io/AVT/)] [[paper](https://arxiv.org/abs/2106.02036)]

If this code helps with your work, please cite:

R. Girdhar and K. Grauman. **Anticipative Video Transformer.** IEEE/CVF International Conference on Computer Vision (ICCV), 2021.

```
@inproceedings{girdhar2021anticipative,
    title = {{Anticipative Video Transformer}},
    author = {Girdhar, Rohit and Grauman, Kristen},
    booktitle = {ICCV},
    year = 2021
}
```

## Installation

The code was developed on an `Ubuntu 20.04.1 LTS` cluster with `SLURM 20.11.5`,
with NVIDIA DGX servers consisting of 8 V100 16GB GPUs.

First set up the required packages in a conda environment. You might need to
make minor modifications here if some packages are no longer available. In
most cases they should be replaceable by more recent versions.
```
$ conda env create -f env.yaml python=3.7.7
$ conda activate avt
```

## Datasets

