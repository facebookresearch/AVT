
## EPIC-Kitchens-100 Test/challenge submission

Any of the models can be trained/tested on train+val/test by changing the
`dataset@dataset_train` and `dataset@dataset_eval` fields in the configs.
Here we provide the configs that were used for the challenge submission.

| Backbone | Head | Train data |  Config | Model |
|----------|------|--------|-------|-------|
| TSN (RGB) | RULSTM | train |  `expts/05_ek100_rustm_test_testonly.txt` | [link](https://iplab.dmi.unict.it/sharing/rulstm/ek100_models/RULSTM-anticipation_0.25_6_8_rgb_mt5r_best.pth.tar) |
| TSN (RGB) | AVT-h | train | `expts/02_ek100_avt_tsn_test_testonly.txt` | [link](https://dl.fbaipublicfiles.com/avt/checkpoints/expts/02_ek100_avt_tsn.txt/0/checkpoint.pth) |
| TSN (RGB) | AVT-h | train + val | `expts/02_ek100_avt_tsn_test_trainval.txt` | [link](https://dl.fbaipublicfiles.com/avt/checkpoints/expts/02_ek100_avt_tsn_test_trainval.txt/0/checkpoint.pth) |
| irCSN-152 (IG65M) | AVT-h | train | `expts/04_ek100_avt_ig65m_test_testonly.txt` | [link](https://dl.fbaipublicfiles.com/avt/checkpoints/expts/04_ek100_avt_ig65m.txt/0/checkpoint.pth) |
| irCSN-152 (IG65M) | AVT-h | train + val | `expts/04_ek100_avt_ig65m_test_trainval.txt` | [link](https://dl.fbaipublicfiles.com/avt/checkpoints/expts/04_ek100_avt_ig65m_test_trainval.txt/0/checkpoint.pth) |
| AVT-b (RGB) | AVT-h | train | `expts/01_ek100_avt_test_testonly.txt` | [link](https://dl.fbaipublicfiles.com/avt/checkpoints/expts/01_ek100_avt.txt/0/checkpoint.pth) |
| AVT-b (RGB) | AVT-h | train + val | `expts/01_ek100_avt_test_trainval.txt` | [link](https://dl.fbaipublicfiles.com/avt/checkpoints/expts/01_ek100_avt_test_trainval.txt/0/checkpoint.pth) |
| TSN (Flow) | AVT-h | train | `expts/06_ek100_avt_tsnflow_test_testonly.txt` | [link](https://dl.fbaipublicfiles.com/avt/checkpoints/expts/06_ek100_avt_tsnflow.txt/0/checkpoint.pth) |
| TSN (Flow) | AVT-h | train + val | `expts/06_ek100_avt_tsnflow_test_trainval.txt` | [link](https://dl.fbaipublicfiles.com/avt/checkpoints/expts/06_ek100_avt_tsnflow_test_trainval.txt/0/checkpoint.pth) |
| TSN (Obj) | AVT-h | train + val | `expts/03_ek100_avt_tsn_obj_test_trainval.txt` | [link](https://dl.fbaipublicfiles.com/avt/checkpoints/expts/03_ek100_avt_tsn_obj_test_trainval.txt/0/checkpoint.pth) |
| AVT-b (RGB, longer) | AVT-h | train | `expts/07_ek100_avt_longer_test_testonly.txt` | [link](https://dl.fbaipublicfiles.com/avt/checkpoints/expts/07_ek100_avt_longer.txt/0/checkpoint.pth) |
| AVT-b (RGB, longer) | AVT-h | train + val | `expts/07_ek100_avt_longer_test_trainval.txt` | [link](https://dl.fbaipublicfiles.com/avt/checkpoints/expts/07_ek100_avt_longer_test_trainval.txt/0/checkpoint.pth) |



The predictions from all the above models were late fused and submitted
for evaluation using the following script:


```python
from notebooks.utils import *
CFG_FILES = [
    # RULSTM
    ('expts/05_ek100_rustm_test_testonly.txt', 0),
    # TSN + AVT-h (train and train+val models)
    ('expts/02_ek100_avt_tsn_test_testonly.txt', 0),
    ('expts/02_ek100_avt_tsn_test_trainval.txt', 0),
    # irCSN152/IG65M + AVT-h
    ('expts/04_ek100_avt_ig65m_test_testonly.txt', 0),
    ('expts/04_ek100_avt_ig65m_test_trainval.txt', 0),
    # AVT
    ('expts/01_ek100_avt_test_testonly.txt', 0),
    ('expts/01_ek100_avt_test_trainval.txt', 0),
    # Flow, obj AVT
    ('expts/06_ek100_avt_tsnflow_test_testonly.txt', 0),
    ('expts/06_ek100_avt_tsnflow_test_trainval.txt', 0),
    ('expts/03_ek100_avt_tsn_obj_test_trainval.txt', 0),
    # Longer AVT
    ('expts/07_ek100_avt_longer_test_testonly.txt', 0),
    ('expts/07_ek100_avt_longer_test_trainval.txt', 0),

]
WTS = [1.0, # RULSTM
       # TSN + AVT-h
       1.0, 1.0,
       # irCSN152/IG65M + AVT-h
       1.0, 1.0,
       # AVT
       0.5, 0.5,
       # Flow, obj AVT
       0.5, 0.5, 0.5,
       # Longer AVT
       1.5, 1.5]
SLS = [2, 4, 4]

package_results_for_submission_ek100(CFG_FILES, WTS, SLS)
```

It should obtain 16.74 on the challenge leaderboard. We also provide our
final submission file [here](https://dl.fbaipublicfiles.com/avt/challenge_submissions/ek100.zip).

## EPIC-Kitchens-55

| Backbone | Head | Top-1 | Top-5 | Config (for top-1/5) | Model (for top-1/5) | AR5 | Config (for AR5) | Model (for AR5) |
|----------|------|------|--------|--------|-----|-----|-----|----|
| TSN (RGB) | AVT-h | 13.1 | 28.1 | `expts/08_ek55_avt_tsn.txt` | [link](https://dl.fbaipublicfiles.com/avt/checkpoints/expts/08_ek55_avt_tsn.txt/0/checkpoint.pth)| 13.5 | `expts/08_ek55_avt_tsn_forAR.txt` | [link](https://dl.fbaipublicfiles.com/avt/checkpoints/expts/08_ek55_avt_tsn_forAR.txt/0/checkpoint.pth) |
| AVT-b | AVT-h | 12.5 | 30.1 | `expts/09_ek55_avt.txt` | [link](https://dl.fbaipublicfiles.com/avt/checkpoints/expts/09_ek55_avt.txt/0/checkpoint.pth)| 13.6 | `expts/09_ek55_avt_forAR.txt` | [link](https://dl.fbaipublicfiles.com/avt/checkpoints/expts/09_ek55_avt_forAR.txt/0/checkpoint.pth) |
| irCSN-152 (IG65M) | AVT-h | 14.4 | 31.7 | `expts/10_ek55_avt_ig65m.txt` | [link](https://dl.fbaipublicfiles.com/avt/checkpoints/expts/10_ek55_avt_ig65m.txt/0/checkpoint.pth)| 13.2 | `expts/10_ek55_avt_ig65m_forAR.txt` | [link](https://dl.fbaipublicfiles.com/avt/checkpoints/expts/10_ek55_avt_ig65m_forAR.txt/0/checkpoint.pth) |

Our final test submission was generated by late-fusing AVT model with predictions from [prior work](https://arxiv.org/abs/2006.00830), and is available [here](https://dl.fbaipublicfiles.com/avt/challenge_submissions/ek55.zip).

## EGTEA Gaze+

| Backbone | Head | Top-1 (Act) | Class-mean (Act) | Config | Model |
|----------|------|-------------|------------------|-------|-------|
| TSN (RGB) | AVT-h | 39.8 | 28.3 | `expts/11_egtea_avt_tsn.txt` | [link](https://dl.fbaipublicfiles.com/avt/checkpoints/expts/11_egtea_avt_tsn.txt/0/checkpoint.pth) |
| AVT-b | AVT-h | 43.0 | 35.2 | `expts/12_egtea_avt.txt` | [link](https://dl.fbaipublicfiles.com/avt/checkpoints/expts/12_egtea_avt.txt/0/checkpoint.pth) |


## 50-Salads

| Backbone | Head | Top-1 (Act) | Config | Model |
|----------|------|-------------|-------|-------|
| AVT-b | AVT-h | 48.0 | `expts/13_50s_avt.txt` | [fold 1](https://dl.fbaipublicfiles.com/avt/checkpoints/expts/13_50s_avt.txt/0/checkpoint.pth)  [fold 2](https://dl.fbaipublicfiles.com/avt/checkpoints/expts/13_50s_avt.txt/1/checkpoint.pth)  [fold 3](https://dl.fbaipublicfiles.com/avt/checkpoints/expts/13_50s_avt.txt/2/checkpoint.pth) [fold 4](https://dl.fbaipublicfiles.com/avt/checkpoints/expts/13_50s_avt.txt/3/checkpoint.pth) [fold 5](https://dl.fbaipublicfiles.com/avt/checkpoints/expts/13_50s_avt.txt/4/checkpoint.pth) |
