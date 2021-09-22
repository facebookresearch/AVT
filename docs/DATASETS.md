## EGTEA Gaze+

The annotations are used from RULSTM, so you'd need to set it up as described in the main README.

### To train models on pre-extracted TSN features

1. Download the features from [RULSTM](https://iplab.dmi.unict.it/sharing/rulstm/features/egtea.zip)
2. Unzip into `DATA/external/rulstm/RULSTM/egtea/`

### To train models on raw videos

Download the videos from [here](https://www.dropbox.com/s/uwwj6wb1j4rsm02/video_links.txt) into `DATA/videos/EGTEA/101020/videos/`

## 50-Salads

1. Download videos from [here](https://cvip.computing.dundee.ac.uk/datasets/foodpreparation/50salads/data/) into `DATA/videos/50Salads/`.
2. Download annotations
   - The models in this paper use annotations from [here](https://github.com/yabufarha/anticipating-activities/issues/5#issuecomment-555916894)
      - Download the `cvpr18_data` folder to `external/breakfast_50salad_anticipation_annotations/cvpr18_data`
   - Additionally, [this](https://dl.fbaipublicfiles.com/avt/datasets/50salads/annotations.zip) annotations directory was shared by the authors of the above paper as well, download it at `external/breakfast_50salad_anticipation_annotations/annotations/`. Shared here for reproducibility of the code.
