This is a modified version of the [Forecasting-HOI-in-FPV](https://github.com/2020aptx4869lm/Forecasting-Human-Object-Interaction-in-FPV) codebase.
Main edits include:

1. Modifications to compile the depthwise 3D conv kernel on more GPU architectures
2. Minor modifications to video transformation functions to work with AVT code
3. Minor modifications to the metrics code to be used for EGTEA


### Get Started
- Follow INSTALL.md for all dependencies
- Download the datasets and unpack them (see ./tools/unpack_*.sh)
- Re-code the videos if necessary(see ./tools/recode_videos.sh)
- Convert the datasets into our format (see ./tools/convert_dataset.py).
- Download pre-trained models (see this [link](https://drive.google.com/drive/folders/1CYWAwoOYRrub9HTSrcpxLpi4Hb7l_vRQ?usp=sharing))
- Try to train on EPIC-Kitchens. You will need to set up the dataset folders and pre-trained model file in the json config
```shell
python ./train_joint.py ./exp_configs/epic-kitchens/epic-kitchens-action_trainval_i3d_csn152_vae.json --prec_bn
```
- [Optional]: Distributed training on multiple GPUs using synchronized batch norm. You can use the same config file. Learning rate will be automatically re-scaled.
```shell
python -m torch.distributed.launch --nproc_per_node=NUM_GPUS train.py ./exp_configs/epic-kitchens/epic-kitchens-action_trainval_i3d_csn152_vae.json -p 5 --prec_bn --sync_bn
```
- Evaluate a trained model. Models are saved in ./ckpt/your_exp/.
```shell
python ./eval_joint.py ./exp_configs/epic-kitchens/epic-kitchens-action_trainval_i3d_csn152_vae.json --resume your_model
```
- Visualize the training (requires tensorboard)
```shell
tensorboard --logdir ./ckpt/your_exp/logs
```

