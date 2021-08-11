"""Training code."""
from typing import Union, Sequence

import datetime
import os
import time
import sys
import logging
import itertools
import operator
import psutil
import h5py
import subprocess

from tqdm import tqdm
import numpy as np
# Need to import this here, as with pytorch 1.7.1 (or some other CLIP dep)
# it's giving a segmentation fault
# https://github.com/pytorch/pytorch/issues/30651
# Needs to imported before torchvision it seems
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.utils.data
from torch.utils.data.dataloader import default_collate
from torch import nn
import torchvision
import torchvision.datasets.video_utils
from torchvision.datasets.samplers import (DistributedSampler,
                                           UniformClipSampler,
                                           RandomClipSampler)
import torch.distributed as dist
import hydra
from omegaconf import OmegaConf

from models import base_model
from common import scheduler, utils, transforms as T
from common.log import MetricLogger, setup_tbx, get_default_loggers
from datasets.data import get_dataset
from external import forecasting_hoi_datasets
from notebooks import utils as nb_utils
from func.train_eval_ops import DenseRegressionLossAccuracy

try:
    from apex import amp
except ImportError:
    amp = None

__all__ = ['main', 'evaluate', 'train_one_epoch', 'initial_setup']
RESULTS_SAVE_DIR = 'results'  # Don't put a "/" at the end, will add later
CKPT_FNAME = 'checkpoint.pth'
DATASET_TRAIN_CFG_KEY = 'dataset_train'
DATASET_EVAL_CFG_KEY = 'dataset_eval'
STR_UID_MAXLEN = 64  # Max length of the string UID stored in H5PY


def store_checkpoint(fpaths: Union[str, Sequence[str]], model, optimizer,
                     lr_scheduler, epoch):
    """
    Args:
        fpaths: List of paths or a single path, where to store.
        model: the model to be stored
        optimizer, lr_scheduler
        epoch: How many epochs have elapsed when this model is being stored.
    """
    model_without_ddp = model
    if isinstance(model, nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    checkpoint = {
        "model": model_without_ddp.state_dict(),
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": lr_scheduler.state_dict(),
        "epoch": epoch,
    }
    if not isinstance(fpaths, list):
        fpaths = [fpaths]
    for fpath in fpaths:
        logging.info('Storing ckpt at epoch %f to %s', epoch, fpath)
        utils.save_on_master(checkpoint, fpath)


def _store_video_logs(data, key, step_id, print_large_freq, metric_logger):
    """
    Args:
        data[key] -> video (B, #clips, 3, T, H, W)
    """
    if metric_logger.writer is None:
        return
    if step_id % print_large_freq != 0:
        return
    if key not in data:
        return
    video = data[key]
    if video.ndim != 6:
        return
    ## Store the videos
    # Swap dims to get N*#clips,T,C,H,W format used by tensorboard
    video = torch.flatten(video, 0, 1)
    vid_log = torch.transpose(video, 1, 2)
    vid_log = vid_log - vid_log.min()
    vid_log = vid_log / vid_log.max()
    kwargs = {}
    if 'video_info' in data:
        # Can't specify different frame rate for videos, so use the min
        kwargs['fps'] = max(
            data['video_info']['video_fps'].min().cpu().numpy().tolist(), 4)
    # TODO(rgirdhar): If a label text is available, overlay on the
    # video. That will help make sure the labels are correct
    metric_logger.writer.add_video(key, vid_log, step_id, **kwargs)


def _store_scalar_logs(name, val, step_id, print_freq, metric_logger):
    if metric_logger.writer is None:
        return
    if step_id % print_freq != 0:
        return
    metric_logger.writer.add_scalar(name, val, step_id)


def _get_memory_usage_gb():
    mem = psutil.virtual_memory()
    return mem.used / (1024**3)


def _compute_final_acc_from_stored(results_dir, dataset):
    results = nb_utils.read_results(os.getcwd(), '', results_dir)
    accs = {}
    for key in results.keys():
        if not key.startswith('logits/'):
            continue
        base_key = key[len('logits/'):]
        top1, top5, ar5, top1_meancls, _ = nb_utils.compute_accuracy(
            results[key], results[f'target/{base_key}'])
        _, _, ar5_ms, _, _ = nb_utils.compute_accuracy(
            results[key], results[f'target/{base_key}'],
            dataset.classes_manyshot[base_key])
        accs[f'final_acc/{base_key}/top1'] = top1
        accs[f'final_acc/{base_key}/top1_meanOverClasses'] = top1_meancls
        accs[f'final_acc/{base_key}/top5'] = top5
        accs[f'final_acc/{base_key}/AR5'] = ar5
        accs[f'final_acc/{base_key}/AR5_manyshot'] = ar5_ms
    return accs


def train_one_epoch(
        train_eval_op,
        optimizer,
        lr_scheduler,
        data_loader,
        epoch: int,
        partial_epoch: float,
        metric_logger,
        logger,
        last_saved_time,
        # kwargs:
        print_freq,
        print_large_freq,
        grad_clip_params,
        loss_wts,  # All the loss wts go here
        save_freq: float,  # num epochs to save at. Could be fractional.
        save_freq_min: float,  # Save a checkpoint every this many minutes
        save_intermediates: bool,
        apex=False,
):
    """
    Args:
        epoch (int) defines how many full epochs have finished
        partial_epoch (float): Defines the ratio of the last epoch that was
            finished before the current model was written out
    """
    header = 'Epoch: [{}]'.format(epoch)
    batches_per_epoch = len(data_loader)
    # Run the data loader for the partial epochs
    partial_iters = int(batches_per_epoch * partial_epoch)
    if partial_iters > 0:
        # TODO: Figure a better way to do this ... too slow
        for i, _ in tqdm(enumerate(data_loader),
                         desc=(f'Loading and throwing data for '
                               f'{partial_epoch:0.8f} epochs or '
                               f'{partial_iters} iters'),
                         total=partial_iters):
            if i >= partial_iters:
                break
    if save_freq:
        save_freq_steps = int(save_freq * batches_per_epoch)
        logger.info('Storing checkpoints every %0.8f epochs, or '
                    '%d steps', save_freq, save_freq_steps)
    if save_freq_min:
        logger.info('Storing checkpoints every %0.2f mins', save_freq_min)
    for i, data in enumerate(
            metric_logger.log_every(data_loader, print_freq, header),
            partial_iters):
        step_id = epoch * batches_per_epoch + i
        cur_epoch = step_id / batches_per_epoch  # Fractional value

        # If provided, store every so many epochs as well. This is useful
        # when doing SSL and want to know how the perf changes at different
        # checkpoint. Storing before so that I can also capture epoch 0.
        # Saving mid-epoch since some pre-training could last days per epoch.
        time_now = datetime.datetime.now()
        mins_since_last_saved = (time_now -
                                 last_saved_time).total_seconds() / 60.0
        if (save_freq and step_id % save_freq_steps == 0) or (
                save_freq_min and (mins_since_last_saved >= save_freq_min)):
            # Not storing in the main checkpoint, keeping that only for the
            # models at full epoch boundaries. So set save_intermediates true
            # to save models at these points
            ckpt_names = []
            if save_intermediates:
                ckpt_names.append(f'checkpoint_ep{cur_epoch:.8f}.pth')
            store_checkpoint(ckpt_names, train_eval_op.model, optimizer,
                             lr_scheduler, cur_epoch)
            last_saved_time = time_now

        start_time = time.time()
        data, _, losses, accuracies = train_eval_op(data, train_mode=True)
        # Reduce the losses, since by default I no longer reduce the losses,
        # to be able to store the outputs
        losses = {key: torch.mean(val) for key, val in losses.items()}
        # Weight the losses
        losses_wtd = []
        for key, val in losses.items():
            this_loss_wt = operator.attrgetter(key)(loss_wts)
            # This will ensure only non 0 loss wts contribute, else otherwise
            # the weight decay will still be associated with this loss.
            if this_loss_wt > 0:
                losses_wtd.append(this_loss_wt * val)
        # Use the total loss to backprop etc
        loss = torch.sum(torch.stack(losses_wtd))
        if torch.isnan(loss):
            raise ValueError('The loss is NaN!')

        optimizer.zero_grad()
        if apex:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        # Clip the gradients if asked for
        if grad_clip_params['max_norm'] is not None:
            params_being_optimized = []
            for param_group in optimizer.param_groups:
                params_being_optimized += param_group['params']
            assert len(params_being_optimized) > 0, 'Shouldnt be training else'
            torch.nn.utils.clip_grad_norm_(params_being_optimized,
                                           **grad_clip_params)

        optimizer.step()

        batch_size = data_loader.batch_size
        metric_logger.update(loss=loss.item(),
                             lr=optimizer.param_groups[0]['lr'])
        metric_logger.meters['clips/s'].update(batch_size /
                                               (time.time() - start_time))

        # Store logs in a sane way
        for acc_key, acc_val in accuracies.items():
            metric_logger.meters[acc_key].update(acc_val.item(), n=batch_size)
        for loss_key, loss_val in losses.items():
            _store_scalar_logs(f'train_per_iter/loss/{loss_key}', loss_val,
                               step_id, print_freq, metric_logger)
        _store_scalar_logs('train_per_iter/loss', loss, step_id, print_freq,
                           metric_logger)
        _store_scalar_logs('train_per_iter/lr',
                           optimizer.param_groups[0]['lr'], step_id,
                           print_freq, metric_logger)
        _store_scalar_logs('train_per_iter/sys/cpu_mem_use_gb',
                           _get_memory_usage_gb(), step_id, print_freq,
                           metric_logger)
        # Store video logs for all videos (future, current etc)
        [
            _store_video_logs(data, key, step_id, print_large_freq,
                              metric_logger) for key in data
            if key.endswith('video')
        ]
        if not isinstance(lr_scheduler.base_scheduler,
                          scheduler.ReduceLROnPlateau):
            # If it is, then that is handled in the main training loop,
            # since it uses the validation accuracy to step down
            lr_scheduler.step()
    return last_saved_time


def store_append_h5(endpoints, output_dir):
    output_fpath = os.path.join(output_dir, f'{utils.get_rank()}.h5')
    os.makedirs(output_dir, exist_ok=True)
    with h5py.File(output_fpath, 'a') as fout:
        for key, val in endpoints.items():
            if key not in fout:
                fout.create_dataset(key,
                                    data=val,
                                    compression='gzip',
                                    compression_opts=9,
                                    chunks=True,
                                    maxshape=(None, ) + val.shape[1:])
            else:
                fout[key].resize((fout[key].shape[0] + val.shape[0], ) +
                                 val.shape[1:])
                fout[key][-val.shape[0]:, ...] = val


def _evaluate_store_logs(logger, metric_logger, acc_keys, store, this_save_dir,
                         data_key, data_loader, epoch, loss_names):
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()

    # gather all accuracies
    final_accuracies = {}
    # Using the loop variable name from earlier .. but ok for now to get
    # the keys
    for acc_key in acc_keys:
        final_accuracies[acc_key] = metric_logger.meters[acc_key].global_avg

    if store:
        # Moving the storage code to the loop, since I am now using it
        # store features which would be too big if waiting till this point
        # store_output = {
        #     k: torch.cat([dic[k] for dic in store_data])
        #     for k in store_data[0]
        # }
        # store_output.update({'epoch': epoch})
        # # Can't really gather on CPU tensorson NCCL, so for now just store
        # # independently in different files
        # this_save_dir = RESULTS_SAVE_DIR + data_key + '/'
        # os.makedirs(this_save_dir, exist_ok=True)
        # torch.save(store_output,
        #            os.path.join(this_save_dir, f'{utils.get_rank()}.pth'))
        dist.barrier()  # all the processes have written out the res
        # Compute the AR@5: will have to read the stored outputs
        final_accuracies.update(
            _compute_final_acc_from_stored(this_save_dir, data_loader.dataset))

    # store logs in a sane way
    for acc_key, acc_val in final_accuracies.items():
        _store_scalar_logs(f'eval_per_epoch{data_key}/{acc_key}', acc_val,
                           int(round(epoch)), 1, metric_logger)
    for loss_name in loss_names:
        _store_scalar_logs(f'eval_per_epoch{data_key}/loss_{loss_name}',
                           metric_logger.meters[loss_name].global_avg,
                           int(round(epoch)), 1, metric_logger)
    logger.info('[{data_key}]'.format(data_key=data_key))
    for key in metric_logger.meters:
        logging.info('%s: %f', key, metric_logger.meters[key].global_avg)
    return final_accuracies


def evaluate(
        train_eval_op,
        data_loaders: dict,
        tb_writer,
        logger,
        epoch: float,  # Can be a partial epoch
        store=True,
        store_endpoint='logits',
        only_run_featext=False):
    """
    Args:
        data_loaders: A dict from key (name) to a data loader. Allows to
            multiple dataloaders for testing on.
        only_run_featext (bool): Set this to true and it will return after the
            features are extracted and won't compute final numbers etc. So
            it will never try to sync processes etc, which leads to crashes.
    """
    all_metric_loggers = {}
    final_accuracies = {}
    for data_key, data_loader in data_loaders.items():
        logger.info('Running evaluation for {0}{1}'.format(
            DATASET_EVAL_CFG_KEY, data_key))
        header = f'[{data_key}] Test:'
        metric_logger = MetricLogger(delimiter='  ',
                                     writer=tb_writer,
                                     stat_set='val' + data_key,
                                     logger=logger)
        all_metric_loggers[data_key] = metric_logger
        this_save_dir = RESULTS_SAVE_DIR + data_key + '/'
        if not only_run_featext:
            # Delete the stored output features files, since with H5 they
            # might be getting appended and will blow up. Note that if
            # feature extraction was the goal and we wanted to append,
            # need to set in the config to not delete the old files so it
            # can append to what has already been computed
            logger.info('Clearing %s/%s/*', os.getcwd(), this_save_dir)
            subprocess.call(f'rm -r {this_save_dir}/*', shell=True)
        for data in metric_logger.log_every(data_loader, 2, header):
            with torch.no_grad():
                data, outputs, losses, accuracies = train_eval_op(
                    data, train_mode=False)
                # Reduce the losses, since by default I no longer reduce the
                # losses, to be able to store the outputs
                losses_reduced = {
                    key: torch.mean(val)
                    for key, val in losses.items()
                }
                loss = torch.sum(torch.stack(list(losses_reduced.values())))
            if store:
                # allow to store logits and logits_regression if that's in too
                all_logits = {
                    key: outputs[key].detach().cpu().numpy()
                    for key in outputs if key.startswith(store_endpoint)
                }
                all_logits.update({'idx': data['idx'].detach().cpu().numpy()})
                uid_data = np.array(data['uid'])
                # If strings, convert format to work with HDF5
                if uid_data.dtype.kind == 'U':
                    # So that it can store upto 64 char strings -- will be
                    # used by the hdf5 too
                    assert int(uid_data.dtype.str[2:]) < STR_UID_MAXLEN, (
                        f'Make sure UID data is smaller than '
                        f'{STR_UID_MAXLEN}, or update that value of '
                        f'STR_UID_MAXLEN')
                    uid_data = uid_data.astype(f'S{STR_UID_MAXLEN}')
                all_logits.update({'uid': uid_data})
                # Storing the actual per batch/elt unreduced losses for
                # potential analysis
                all_logits.update({
                    'loss/' + key: val.detach().cpu()
                    for key, val in losses.items()
                })
                if not only_run_featext:
                    # store the targets as well
                    all_logits.update({
                        'target/' + key: val.detach().cpu().numpy()
                        for key, val in data['target'].items()
                    })
                # Do the actual storage into HDF5s that can append to the
                # stuff from previous batch. Doing it here rather than
                # collecting (as I used to do) so that this can be used
                # for feature extraction where storing into a list will
                # be too expensive
                all_logits.update({'epoch': np.array([epoch])})
                store_append_h5(all_logits, this_save_dir)
            # FIXME need to take into account that the datasets
            # could have been padded in distributed setup
            batch_size = data_loader.batch_size
            metric_logger.update(loss=loss.item())
            for acc_key, acc_val in accuracies.items():
                metric_logger.meters[acc_key].update(acc_val.item(),
                                                     n=batch_size)
            for loss_name, loss_val in losses_reduced.items():
                metric_logger.meters[loss_name].update(loss_val.item(),
                                                       n=batch_size)
        if not only_run_featext:
            final_accuracies[data_key] = _evaluate_store_logs(
                logger, metric_logger, accuracies.keys(), store, this_save_dir,
                data_key, data_loader, epoch, losses_reduced.keys())

    if only_run_featext:
        # None of the rest is needed
        return 0.0

    # Run the following when doing dense prediction, so far I think the best
    # way to check for that is if
    if isinstance(train_eval_op.cls_loss_acc_fn, DenseRegressionLossAccuracy):
        all_moc_scores = nb_utils.compute_dense_moc_all(
            (os.getcwd(), ''), {
                data_key: data_loader.dataset
                for data_key, data_loader in data_loaders.items()
            })
        for data_key, moc_scores_dict in all_moc_scores.items():
            for key, moc_score in moc_scores_dict.items():
                _store_scalar_logs(f'eval_per_epoch{data_key}/moc_{key}',
                                   moc_score, int(round(epoch)), 1,
                                   metric_logger)
    # Return the accuracy on the main evaluation dataset, which must be the
    # one which doesn't have any prefix (i.e. in the dataset_eval)
    # Returning the accuracy metric that is most relevant to the dataset.
    main_dataset_key = ''
    main_metric = final_accuracies[main_dataset_key][
        data_loaders[main_dataset_key].dataset.primary_metric]
    return main_metric


def initial_setup(cfg, logger):
    torchvision.set_video_backend(cfg.pytorch.video_backend)

    if cfg.data_parallel:
        dist_info = {}
        dist_info['distributed'] = False
        dist_info['world_size'] = torch.cuda.device_count()
        # In DDP we set these params for a single process
        cfg.train.batch_size *= dist_info['world_size']
        cfg.eval.batch_size *= dist_info['world_size']
    else:
        dist_info = utils.init_distributed_mode(logger,
                                                dist_backend=cfg.dist_backend)
    logger.info("Dist info:", dist_info)
    logger.info("torch version: %s", torch.__version__)
    logger.info("torchvision version: %s", torchvision.__version__)
    logger.info("hydra version: %s", hydra.__version__)

    device = torch.device('cuda')

    torch.backends.cudnn.benchmark = True
    writer = setup_tbx('logs/', SummaryWriter)
    return dist_info, device, writer


def init_model(model, ckpt_path, modules_to_keep, logger):
    """Initialize model with weights from ckpt_path.
    Args:
        ckpt_path (str): A string with path to file
        modules_to_keep (str): A comma sep string with the module name prefix
            that should be loaded from the checkpoint
    """
    logger.debug('Initing %s with ckpt path: %s, using modules in it %s',
                 model, ckpt_path, modules_to_keep)
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    if 'model' in checkpoint.keys():
        state_dict = checkpoint['model']
    elif 'state_dict' in checkpoint.keys():
        state_dict = checkpoint['state_dict']
    elif 'classy_state_dict' in checkpoint.keys():
        state_dict = checkpoint['classy_state_dict']
        # This is likely coming from a VISSL codebase, so the actual trunk
        # params will be as follows. Ideally support this more generally TODO
        state_dict = state_dict['base_model']['model']['trunk']
    else:
        state_dict = checkpoint
    if modules_to_keep:
        # Keep only the elements of state_dict that match modules to keep.
        # Also, remove that prefix from the names
        filtered_state_dict = {}
        for key, val in state_dict.items():
            for mod_name in modules_to_keep.split(','):
                if key.startswith(mod_name):
                    filtered_state_dict[key[len(mod_name):]] = val
                    continue
        state_dict = filtered_state_dict
    # Ignore any parameters/buffers (bn mean/var) where shape does not match
    for name, param in itertools.chain(model.named_parameters(),
                                       model.named_buffers()):
        if name in state_dict and state_dict[name].shape != param.shape:
            logger.warning('Ckpt shape mismatch for %s (%s vs %s). Ignoring.',
                           name, state_dict[name].shape, param.shape)
            del state_dict[name]
    missing_keys, unexp_keys = model.load_state_dict(state_dict, strict=False)
    logger.warning('Could not init from %s: %s', ckpt_path, missing_keys)
    logger.warning('Unused keys in %s: %s', ckpt_path, unexp_keys)


def collate_fn_remove_audio(batch):
    """Remove audio from the batch.
    Also remove any None(s) -- those were data points I wasn't able to read.
    Not needed, and it doesn't batch properly since it is different length.
    """
    batch = list(filter(lambda x: x is not None, batch))
    if isinstance(batch[0], tuple):
        batch = [(d[0], d[2]) for d in batch]
    return default_collate(batch)


def _get_resize_shape(data_cfg):
    scale_h = data_cfg.scale_h
    scale_w = data_cfg.scale_w
    if isinstance(scale_w, int) and scale_w == -1:
        resize_shape = scale_h
    else:
        assert (not isinstance(scale_h, int) or scale_h != -1), (
            'If using -1, must be used for scale_w. The smaller side will be '
            'scaled by that size.')
        resize_shape = (scale_h, scale_w)
    return resize_shape


def _get_pixel_mean_std(data_cfg):
    return {'mean': tuple(data_cfg.mean), 'std': tuple(data_cfg.std)}


def _set_all_bn_to_not_track_running_mean(model):
    """
    Set all batch norm layers to not use running mean.
    """
    for module in model.modules():
        # This should be able to capture any BatchNorm1d, 2d, 3d etc.
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.track_running_stats = False
    return model


def main(cfg):
    logger = logging.getLogger(__name__)
    dist_info, device, writer = initial_setup(cfg, logger)
    # Data loading code
    logger.info("Loading data")

    logger.info("\t Loading datasets")
    st = time.time()

    # separate these into get transforms
    # TODO: This is gotten too complex: clean up, make interface better
    transform_train = [
        T.ToTensorVideo(),
        T.Resize(_get_resize_shape(cfg.data_train)),
        T.RandomHorizontalFlipVideo(cfg.data_train.flip_p),
        T.ColorJitterVideo(brightness=cfg.data_train.color_jitter_brightness,
                           contrast=cfg.data_train.color_jitter_contrast,
                           saturation=cfg.data_train.color_jitter_saturation,
                           hue=cfg.data_train.color_jitter_hue),
        torchvision.transforms.Lambda(
            lambda x: x * cfg.data_train.scale_pix_val),
        torchvision.transforms.Lambda(lambda x: x[[2, 1, 0], ...])
        if cfg.data_train.reverse_channels else torchvision.transforms.Compose(
            []),
        T.NormalizeVideo(**_get_pixel_mean_std(cfg.data_train)),
    ]
    if cfg.data_train.crop_size is not None:
        transform_train.append(
            T.RandomCropVideo(
                (cfg.data_train.crop_size, cfg.data_train.crop_size)), )
    # Insert Miao's transforms if needed. Not doing by default since that is
    # untested code and don't want all images to pass through it if not
    # necessary
    if cfg.data_train.color_jitter_miao > 0:
        transform_train.insert(
            0,
            forecasting_hoi_datasets.VideoRandomColor(
                cfg.data_train.color_jitter_miao))
    if cfg.data_train.rotate_miao > 0:
        transform_train.insert(
            0,
            forecasting_hoi_datasets.VideoRandomRotate(
                cfg.data_train.rotate_miao))
    transform_train = torchvision.transforms.Compose(transform_train)
    transform_eval = [
        T.ToTensorVideo(),
        T.Resize(_get_resize_shape(cfg.data_eval)),
        torchvision.transforms.Lambda(
            lambda x: x * cfg.data_eval.scale_pix_val),
        torchvision.transforms.Lambda(lambda x: x[[2, 1, 0], ...]) if
        cfg.data_eval.reverse_channels else torchvision.transforms.Compose([]),
        T.NormalizeVideo(**_get_pixel_mean_std(cfg.data_eval)),
    ]
    if cfg.data_eval.crop_size is not None:
        transform_eval.append(
            T.MultiCropVideo(
                (cfg.data_eval.crop_size, cfg.data_eval.crop_size),
                cfg.data_eval.eval_num_crops, cfg.data_eval.eval_flip_crops))
    transform_eval = torchvision.transforms.Compose(transform_eval)

    datasets_train = [
        get_dataset(getattr(cfg, el), cfg.data_train, transform_train, logger)
        for el in cfg.keys() if el.startswith(DATASET_TRAIN_CFG_KEY)
    ]
    if len(datasets_train) > 1:
        dataset = torch.utils.data.ConcatDataset(datasets_train)
    else:
        dataset = datasets_train[0]
    # could be multiple test datasets
    datasets_test = {
        el[len(DATASET_EVAL_CFG_KEY):]:
        get_dataset(getattr(cfg, el), cfg.data_eval, transform_eval, logger)
        for el in cfg.keys() if el.startswith(DATASET_EVAL_CFG_KEY)
    }

    logger.info("Took %d", time.time() - st)

    logger.info("Creating data loaders")
    train_sampler = None
    test_samplers = {key: None for key in datasets_test}
    if hasattr(dataset, 'video_clips'):
        assert cfg.train.shuffle_data, 'TODO'
        train_sampler = RandomClipSampler(getattr(dataset, 'video_clips'),
                                          cfg.data_train.train_bs_multiplier)
        test_samplers = {
            key: UniformClipSampler(val.video_clips,
                                    cfg.data_eval.val_clips_per_video)
            for key, val in datasets_test.items()
        }
        if dist_info['distributed']:
            train_sampler = DistributedSampler(train_sampler)
            test_samplers = [DistributedSampler(el) for el in test_samplers]
    elif dist_info['distributed']:
        # Distributed, but doesn't have video_clips
        if cfg.data_train.use_dist_sampler:
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                dataset,
                num_replicas=dist_info['world_size'],
                rank=dist_info['rank'],
                shuffle=cfg.train.shuffle_data)
        if cfg.data_eval.use_dist_sampler:
            test_samplers = {
                key: torch.utils.data.distributed.DistributedSampler(
                    val,
                    num_replicas=dist_info['world_size'],
                    rank=dist_info['rank'],
                    shuffle=False)
                for key, val in datasets_test.items()
            }

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.train.batch_size,
        sampler=train_sampler,
        num_workers=cfg.data_train.workers,
        pin_memory=False,  # usually hurts..
        shuffle=(train_sampler is None and cfg.train.shuffle_data),
        collate_fn=collate_fn_remove_audio,
    )

    data_loaders_test = {
        key: torch.utils.data.DataLoader(
            val,
            # Since no backprop, so can have a larger batch size
            batch_size=cfg.eval.batch_size or cfg.train.batch_size * 4,
            sampler=test_samplers[key],
            num_workers=cfg.data_eval.workers,
            pin_memory=False,  # Usually hurts..
            shuffle=False,
            collate_fn=collate_fn_remove_audio,
        )
        for key, val in datasets_test.items()
    }

    num_classes = {key: len(val) for key, val in dataset.classes.items()}
    logger.info('Creating model with %s classes', num_classes)
    model = base_model.BaseModel(cfg.model,
                                 num_classes=num_classes,
                                 class_mappings=dataset.class_mappings)
    logger.debug('Model: %s', model)
    if dist_info['distributed'] and cfg.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    if cfg.train.init_from_model:
        # This can have structure as follows:
        # <module name>:<path to init model>;<module name>:<path>: ...
        for module_ckpt in cfg.train.init_from_model:
            elts = module_ckpt
            if len(elts) == 1:
                model_to_init = model
                ckpt_modules_to_keep = None
                ckpt_path = elts[0]
            elif len(elts) == 2:
                model_to_init = operator.attrgetter(elts[0])(model)
                ckpt_modules_to_keep = None
                ckpt_path = elts[1]
            elif len(elts) == 3:
                model_to_init = operator.attrgetter(elts[0])(model)
                ckpt_modules_to_keep = elts[1]
                ckpt_path = elts[2]
            else:
                raise ValueError(f'Incorrect formatting {module_ckpt}')
            init_model(model_to_init, ckpt_path, ckpt_modules_to_keep, logger)

    model.to(device)

    if cfg.opt.classifier_only:
        assert len(cfg.opt.lr_wd) == 1
        assert cfg.opt.lr_wd[0][0] == 'classifier'
        model = _set_all_bn_to_not_track_running_mean(model)
    params = []
    for this_module_names, this_lr, this_wd in cfg.opt.lr_wd:
        if OmegaConf.get_type(this_module_names) != list:
            this_module_names = [this_module_names]
        this_modules = [
            operator.attrgetter(el)(model) if el != '__all__' else model
            for el in this_module_names
        ]
        this_params_bias_bn = {}
        this_params_rest = {}
        for this_module_name, this_module in zip(this_module_names,
                                                 this_modules):
            for name, param in this_module.named_parameters():
                # ignore the param without grads
                if not param.requires_grad:
                    continue
                # May not always have a ".bias" if it's the last element, and no
                # module name
                if name.endswith('bias') or ('.bn' in name):
                    this_params_bias_bn[this_module_name + '.' + name] = param
                else:
                    this_params_rest[this_module_name + '.' + name] = param
        this_scaled_lr = this_lr * dist_info['world_size']
        if cfg.opt.scale_lr_by_bs:
            this_scaled_lr *= cfg.train.batch_size
        params.append({
            'params': this_params_rest.values(),
            'lr': this_scaled_lr,
            'weight_decay': this_wd,
        })
        logger.info('Using LR %f WD %f for parameters %s', params[-1]['lr'],
                    params[-1]['weight_decay'], this_params_rest.keys())
        params.append({
            'params': this_params_bias_bn.values(),
            'lr': this_scaled_lr,
            'weight_decay': this_wd * cfg.opt.bias_bn_wd_scale,
        })
        logger.info('Using LR %f WD %f for parameters %s', params[-1]['lr'],
                    params[-1]['weight_decay'], this_params_bias_bn.keys())
    # Remove any parameters for which LR is 0; will save GPU usage
    params_final = []
    for param_lr in params:
        if param_lr['lr'] != 0.0:
            params_final.append(param_lr)
        else:
            for param in param_lr['params']:
                param.requires_grad = False

    optimizer = hydra.utils.instantiate(cfg.opt.optimizer, params_final)

    if cfg.apex:
        model, optimizer = amp.initialize(model,
                                          optimizer,
                                          opt_level=cfg.apex_opt_level)

    # convert scheduler to be per iteration,
    # not per epoch, for warmup that lasts
    # between different epochs
    main_scheduler = hydra.utils.instantiate(
        cfg.opt.scheduler,
        optimizer,
        iters_per_epoch=len(data_loader),
        world_size=dist_info['world_size'])
    lr_scheduler = hydra.utils.instantiate(cfg.opt.warmup,
                                           optimizer,
                                           main_scheduler,
                                           iters_per_epoch=len(data_loader),
                                           world_size=dist_info['world_size'])

    last_saved_ckpt = CKPT_FNAME
    start_epoch = 0
    if os.path.isfile(last_saved_ckpt):
        checkpoint = torch.load(last_saved_ckpt, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        start_epoch = checkpoint['epoch']
        logger.warning('Loaded model from %s (ep %f)', last_saved_ckpt,
                       start_epoch)

    if dist_info['distributed'] and not cfg.eval.eval_fn.only_run_featext:
        # If only feat ext, then each gpu is going to test separately anyway,
        # no need for communication between the models
        logger.info('Wrapping model into DDP')
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[dist_info['gpu']],
            output_device=dist_info['gpu'])
    elif cfg.data_parallel:
        logger.info('Wrapping model into DP')
        device_ids = range(dist_info['world_size'])
        model = torch.nn.parallel.DataParallel(model, device_ids=device_ids)

    # TODO add an option here to support val mode training
    # Passing in the training dataset, since that will be used for computing
    # weights for classes etc.
    train_eval_op = hydra.utils.instantiate(cfg.train_eval_op,
                                            model,
                                            device,
                                            dataset,
                                            _recursive_=False)

    if cfg.test_only:
        logger.info("Starting test_only")
        hydra.utils.call(cfg.eval.eval_fn, train_eval_op, data_loaders_test,
                         writer, logger, start_epoch)
        return

    logger.info("Start training")
    start_time = time.time()

    # Get training metric logger
    stat_loggers = get_default_loggers(writer, start_epoch, logger)
    best_acc1 = 0.0
    partial_epoch = start_epoch - int(start_epoch)
    start_epoch = int(start_epoch)
    last_saved_time = datetime.datetime(1, 1, 1, 0, 0)
    epoch = 0  # Since using this var to write the checkpoint output, so init to sth
    for epoch in range(start_epoch, cfg.train.num_epochs):
        if dist_info['distributed'] and train_sampler is not None:
            train_sampler.set_epoch(epoch)
        last_saved_time = hydra.utils.call(cfg.train.train_one_epoch_fn,
                                           train_eval_op, optimizer,
                                           lr_scheduler, data_loader, epoch,
                                           partial_epoch,
                                           stat_loggers["train"], logger,
                                           last_saved_time)
        partial_epoch = 0  # Reset, for future epochs
        store_checkpoint([CKPT_FNAME], model, optimizer, lr_scheduler,
                         epoch + 1)
        if cfg.train.eval_freq and epoch % cfg.train.eval_freq == 0:
            acc1 = hydra.utils.call(cfg.eval.eval_fn, train_eval_op,
                                    data_loaders_test, writer, logger,
                                    epoch + 1)
        else:
            acc1 = 0
        if cfg.train.store_best and acc1 >= best_acc1:
            store_checkpoint('checkpoint_best.pth', model, optimizer,
                             lr_scheduler, epoch + 1)
            best_acc1 = acc1

        if isinstance(lr_scheduler.base_scheduler,
                      scheduler.ReduceLROnPlateau):
            lr_scheduler.step(acc1)

        # reset all meters in the metric logger
        for log in stat_loggers:
            stat_loggers[log].reset_meters()
    # Store the final model to checkpoint
    store_checkpoint([CKPT_FNAME], model, optimizer, lr_scheduler, epoch + 1)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time %s', total_time_str)
