# Anticipative Video Transformer

<p><img src="https://rohitgirdhar.github.io/DetectAndTrack/assets/cup.png" width="30px" align="center" /> Ranked <b>first</b> in the Action Anticipation task of the <a href="https://epic-kitchens.github.io/2021#results">CVPR 2021 EPIC-Kitchens Challenge</a>! (entry: AVT-FB-UT)</p>

<img src="https://facebookresearch.github.io/AVT/assets/avt.gif" width="50%" />

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/anticipative-video-transformer/action-anticipation-on-epic-kitchens-100)](https://paperswithcode.com/sota/action-anticipation-on-epic-kitchens-100?p=anticipative-video-transformer) <br/>
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/anticipative-video-transformer/action-anticipation-on-epic-kitchens-100-test)](https://paperswithcode.com/sota/action-anticipation-on-epic-kitchens-100-test?p=anticipative-video-transformer) <br/>
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/anticipative-video-transformer/action-anticipation-on-epic-kitchens-55-seen)](https://paperswithcode.com/sota/action-anticipation-on-epic-kitchens-55-seen?p=anticipative-video-transformer) <br/>
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/anticipative-video-transformer/action-anticipation-on-epic-kitchens-55-1)](https://paperswithcode.com/sota/action-anticipation-on-epic-kitchens-55-1?p=anticipative-video-transformer) <br/>




[[project page](https://facebookresearch.github.io/AVT/)] [[paper](https://arxiv.org/abs/2106.02036)]

If this code helps with your work, please cite:

R. Girdhar and K. Grauman. **Anticipative Video Transformer.** IEEE/CVF International Conference on Computer Vision (ICCV), 2021.

```bibtex
@inproceedings{girdhar2021anticipative,
    title = {{Anticipative Video Transformer}},
    author = {Girdhar, Rohit and Grauman, Kristen},
    booktitle = {ICCV},
    year = 2021
}
```

## Installation

The code was tested on a `Ubuntu 20.04` cluster
with each server consisting of 8 V100 16GB GPUs.

First clone the repo and set up the required packages in a conda environment.
You might need to make minor modifications here if some packages are no longer
available. In most cases they should be replaceable by more recent versions.

```bash
$ git clone --recursive git@github.com:facebookresearch/AVT.git
$ conda env create -f env.yaml python=3.7.7
$ conda activate avt
```

### Set up RULSTM codebase

If you plan to use EPIC-Kitchens datasets,
you might need the train/test splits and evaluation code from RULSTM. This is also needed
if you want to extract RULSTM predictions for test submissions.

```bash
$ cd external
$ git clone git@github.com:fpv-iplab/rulstm.git; cd rulstm
$ git checkout 57842b27d6264318be2cb0beb9e2f8c2819ad9bc
$ cd ../..
```

## Datasets

The code expects the data in the `DATA/` folder. You can also symlink it to
a different folder on a faster/larger drive. Inside it will contain following folders:
1) `videos/` which will contain raw videos
2) `external/` which will contain pre-extracted features from prior work
3) `extracted_features/` which will contain other extracted features
4) `pretrained/` which contains pretrained models, eg from TIMM

The paths to these datasets are set
in files like [`conf/dataset/epic_kitchens100/common.yaml`](conf/dataset/epic_kitchens100/common.yaml)
so you can also update the paths there instead.

### EPIC-Kitchens

To train only the AVT-h on top of pre-extracted features, you can download the
features from RULSTM into `DATA/external/rulstm/RULSTM/data_full` for [EK55](https://github.com/fpv-iplab/rulstm/blob/master/RULSTM/scripts/download_data_ek55_full.sh) and
`DATA/external/rulstm/RULSTM/ek100_data_full`
for [EK100](https://github.com/fpv-iplab/rulstm/blob/master/RULSTM/scripts/download_data_ek100_full.sh).
If you plan to train models on features extracted from a irCSN-152 model
finetuned from IG65M features, you can download our pre-extracted features
from [here](https://dl.fbaipublicfiles.com/avt/datasets/ek100/ig65m_ftEk100_logits_10fps1s/rgb/data.mdb) into `DATA/extracted_features/ek100/ig65m_ftEk100_logits_10fps1s/rgb/` or [here](https://dl.fbaipublicfiles.com/avt/datasets/ek55/ig65m_ftEk55train_logits_25fps/rgb/data.mdb) into `DATA/extracted_features/ek55/ig65m_ftEk55train_logits_25fps/rgb/`.

To train AVT end-to-end, you need to download the raw videos from [EPIC-Kitchens](https://data.bris.ac.uk/data/dataset/2g1n6qdydwa9u22shpxqzp0t8m). They can be organized as you wish, but this
is how my folders are organized (since I first downloaded EK55 and then the remaining
new videos for EK100):

```
DATA
├── videos
│   ├── EpicKitchens
│   │   └── videos_ht256px
│   │       ├── train
│   │       │   ├── P01
│   │       │   │   ├── P01_01.MP4
│   │       │   │   ├── P01_03.MP4
│   │       │   │   ├── ...
│   │       └── test
│   │           ├── P01
│   │           │   ├── P01_11.MP4
│   │           │   ├── P01_12.MP4
│   │           │   ├── ...
│   │           ...
│   ├── EpicKitchens100
│   │   └── videos_extension_ht256px
│   │       ├── P01
│   │       │   ├── P01_101.MP4
│   │       │   ├── P01_102.MP4
│   │       │   ├── ...
│   │       ...
│   ├── EGTEA/101020/videos/
│   │   ├── OP01-R01-PastaSalad.mp4
│   │   ...
│   └── 50Salads/rgb/
│       ├── rgb-01-1.avi
│       ...
├── external
│   └── rulstm
│       └── RULSTM
│           ├── egtea
│           │   ├── TSN-C_3_egtea_action_CE_flow_model_best_fcfull_hd
│           │   ...
│           ├── data_full  # (EK55)
│           │   ├── rgb
│           │   ├── obj
│           │   └── flow
│           └── ek100_data_full
│               ├── rgb
│               ├── obj
│               └── flow
└── extracted_features
    ├── ek100
    │   └── ig65m_ftEk100_logits_10fps1s
    │       └── rgb
    └── ek55
        └── ig65m_ftEk55train_logits_25fps
            └── rgb
```

If you use a different organization, you would need to edit the train/val
dataset files, such as [`conf/dataset/epic_kitchens100/anticipation_train.yaml`](conf/dataset/epic_kitchens100/anticipation_train.yaml). Sometimes the values are overriden
in the TXT config files, so might need to change there too. The `root` property takes a list of
folders where the videos can be found, and it will search through all of them
in order for a given video. Note that we resized the EPIC videos to
256px height for faster processing; you can use [`sample_scripts/resize_epic_256px.sh`](sample_scripts/resize_epic_256px.sh) script for the same.

Please see [`docs/DATASETS.md`](docs/DATASETS.md) for setting up other datasets.

## Training and evaluating models

If you want to train AVT models, you would need pre-trained models from
[`timm`](https://github.com/rwightman/pytorch-image-models/tree/8257b86550b8453b658e386498d4e643d6bf8d38).
We have experiments that use the following models:

```bash
$ mkdir DATA/pretrained/TIMM/
$ wget https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_patch16_224_in21k-e5005f0a.pth -O DATA/pretrained/TIMM/jx_vit_base_patch16_224_in21k-e5005f0a.pth
$ wget https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth -O DATA/pretrained/TIMM/jx_vit_base_p16_224-80ecf9dd.pth
```

The code uses [`hydra 1.0`](https://hydra.cc/) for configuration with [`submitit`](https://github.com/facebookincubator/submitit) plugin for jobs
via SLURM. We provide a `launch.py` script that is a wrapper around the
training scripts and can run jobs locally or launch distributed jobs. The
configuration overrides for a specific experiment is defined by a TXT file.
You can run a config by:

```bash
$ python launch.py -c expts/01_ek100_avt.txt
```
where `expts/01_ek100_avt.txt` can be replaced by any TXT config file.

By default, the launcher will launch the job to a SLURM cluster. However,
you can run it locally using one of the following options:

1. `-g` to run locally in debug mode with 1 GPU and 0 workers. Will allow you to place
`pdb.set_trace()` to debug interactively.
2. `-l` to run locally using as many GPUs on the local machine.

This will run the training, which will run validation every few epochs. You can
also only run testing using the `-t` flag.

The outputs will be stored in `OUTPUTS/<path to config>`. This would include
tensorboard files that you can use to visualize the training progress.

## Model Zoo


### EPIC-Kitchens-100


| Backbone | Head | Class-mean <br/> Recall@5 (Actions) | Config | Model |
|----------|------|-------------------------------|--------|-----|
| AVT-b (IN21K) | AVT-h | 14.9 | `expts/01_ek100_avt.txt` | [link](https://dl.fbaipublicfiles.com/avt/checkpoints/expts/01_ek100_avt.txt/0/checkpoint.pth)|
| TSN (RGB) | AVT-h | 13.6 | `expts/02_ek100_avt_tsn.txt` | [link](https://dl.fbaipublicfiles.com/avt/checkpoints/expts/02_ek100_avt_tsn.txt/0/checkpoint.pth)|
| TSN (Obj) | AVT-h | 8.7 | `expts/03_ek100_avt_tsn_obj.txt` | [link](https://dl.fbaipublicfiles.com/avt/checkpoints/expts/03_ek100_avt_tsn_obj.txt/0/checkpoint.pth)|
| irCSN152 (IG65M) | AVT-h | 12.8 | `expts/04_ek100_avt_ig65m.txt` | [link](https://dl.fbaipublicfiles.com/avt/checkpoints/expts/04_ek100_avt_ig65m.txt/0/checkpoint.pth)|


### Late fusing predictions

For comparison to methods that use multiple modalities, you can late fuse
predictions from multiple models using functions from `notebooks/utils.py`.
For example, to compute the late fused performance reported in Table 3 (val)
as `AVT+` (obtains 15.9 recall@5 for actions):

```python
from notebooks.utils import *
CFG_FILES = [
    ('expts/01_ek100_avt.txt', 0),
    ('expts/03_ek100_avt_tsn_obj.txt', 0),
]
WTS = [2.5, 0.5]
print_accuracies_epic(get_epic_marginalize_late_fuse(CFG_FILES, weights=WTS)[0])
```

Please see [`docs/MODELS.md`](docs/MODELS.md) for test submission and models on other datasets.

## License

This codebase is released under the license terms specified in the [LICENSE](LICENSE) file. Any imported libraries, datasets or other code follows the license terms set by respective authors.


## Acknowledgements

The codebase was built on top of [`facebookresearch/VMZ`](https://github.com/facebookresearch/VMZ). Many thanks to [Antonino Furnari](https://github.com/fpv-iplab/rulstm), [Fadime Sener](https://cg.cs.uni-bonn.de/en/publications/paper-details/sener-2020-temporal/) and [Miao Liu](https://github.com/2020aptx4869lm/Forecasting-Human-Object-Interaction-in-FPV) for help with prior work.
