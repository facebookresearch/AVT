
# Anticipative Video Transformer

<video style="width: 25em; height: 25em; margin: 10px" autoplay muted loop controls>
    <source src="https://facebookresearch.github.io/AVT/assets/videos/6488.mp4" type="video/mp4">
    Your browser does not support the video tag.
</video>
<video style="width: 25em; height: 25em; margin: 10px" autoplay muted loop controls>
    <source src="https://facebookresearch.github.io/AVT/assets/videos/4855.mp4" type="video/mp4">
    Your browser does not support the video tag.
</video>

<p><img src="https://rohitgirdhar.github.io/DetectAndTrack/assets/cup.png" width="30px" align="center" /> Ranked <b>first</b> in the Action Anticipation task of the <a href="https://epic-kitchens.github.io/2021#results">CVPR 2021 EPIC-Kitchens Challenge</a>! (entry: AVT-FB-UT)</p>

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

First clone the repo and set up the required packages in a conda environment.
You might need to make minor modifications here if some packages are no longer
available. In most cases they should be replaceable by more recent versions.
```
$ git clone --recursive git@github.com:facebookresearch/AVT.git
$ conda env create -f env.yaml python=3.7.7
$ conda activate avt
```

## Datasets

The code expects the data in the `DATA/` folder. You can also symlink it to
a different folder on a faster/larger drive. Inside it will contain 2 folders:
1) `videos/` which will contain raw videos; and
2) `external/` which will contain pre-extracted features from prior work.

The paths to these datasets are set
in files like [`conf/dataset/epic_kitchens100/common.yaml`](conf/dataset/epic_kitchens100/common.yaml)
so you can also update the paths there instead.

### EPIC-Kitchens

To train only the AVT-h on top of pre-extracted features, you can download the
features from RULSTM into `DATA/external/rulstm/RULSTM/data_full` for [EK55](https://github.com/fpv-iplab/rulstm/blob/master/RULSTM/scripts/download_data_ek55_full.sh) and
`DATA/external/rulstm/RULSTM/ek100_data_full`
for [EK100](https://github.com/fpv-iplab/rulstm/blob/master/RULSTM/scripts/download_data_ek100_full.sh).

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
└── external
    └── rulstm
        └── RULSTM
            ├── data_full
            │   ├── rgb
            │   ├── obj
            │   └── flow
            └── ek100_data_full
                ├── rgb
                ├── obj
                └── flow
```

If you use a different organization, you would need to edit the train/val
dataset files, such as [`conf/dataset/epic_kitchens100/anticipation_train.yaml`](conf/dataset/epic_kitchens100/anticipation_train.yaml). The `root` property takes a list of
folders where the videos can be found, and it will search through all of them
in order for a given video. Note that we resized the EPIC videos to
256px height for faster processing; you can use [`sample_scripts/resize_epic_256px.sh`](sample_scripts/resize_epic_256px.sh) script for the same.

## Training and evaluating models

The code uses `hydra 1.0` and `submitit` for configuration and launching jobs
via SLURM. We provide a `launch.py` script that is a wrapper around the
training scripts and can run jobs locally or launch distributed jobs.

## Test/challenge submission

## Acknowledgements
