from __future__ import print_function
import datetime
import os
import time
import logging
from functools import partial
import hydra

import torch
import torch.utils.data
from torch import nn
import torchvision
import torchvision.datasets.video_utils

from common import utils, transforms as T
from common.log import get_default_loggers
from common.sampler import DistributedSampler, RandomClipSampler

from external.moco.moco import loader as moco_loader

from datasets.data import get_dataset
from .train import initial_setup


try:
    from apex import amp
except ImportError:
    amp = None

__all__ = ["main"]


def main(cfg):
    logger = logging.getLogger(__name__)
    dist_info, device, writer = initial_setup(cfg, logger)

    # Create model
    logger.info("Creating model")
    model_creator = partial(hydra.utils.instantiate, cfg.model)
    # Wrap around in moco
    model = hydra.utils.instantiate(cfg.moco, model_creator)
    logger.info(model)

    if dist_info['distributed'] and cfg.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model.to(device)
    logger.info('Wrapping model into DDP')
    model_without_ddp = model
    if dist_info['distributed']:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[dist_info['gpu']])
        model_without_ddp = model.module

    criterion = nn.CrossEntropyLoss().cuda(device)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=cfg.opt.lr * dist_info['world_size'],
        momentum=cfg.opt.momentum,
        weight_decay=cfg.opt.weight_decay,
    )

    # Data loading code
    logger.info("Loading data")

    logger.info("\t Loading datasets")
    st = time.time()

    # separate these into get transforms
    transform_train = torchvision.transforms.Compose([
        T.ToTensorVideo(),
        T.Resize((cfg.data.scale_h, cfg.data.scale_w)),
        T.RandomHorizontalFlipVideo(),
        T.NormalizeVideo(mean=(0.43216, 0.394666, 0.37645),
                         std=(0.22803, 0.22145, 0.216989)),
        T.RandomCropVideo((cfg.data.crop_size, cfg.data.crop_size)),
        # TODO(rgirdhar): Add Gaussian blur, color jitter
    ])

    dataset = get_dataset(cfg.dataset, cfg.data,
                          moco_loader.TwoCropsTransform(transform_train))

    logger.info("Took %f", time.time() - st)

    logger.info("Creating data loaders")
    train_sampler = RandomClipSampler(dataset.video_clips,
                                      cfg.data.train_bs_multiplier)
    if dist_info['distributed']:
        train_sampler = DistributedSampler(train_sampler)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.train.batch_size,
        sampler=train_sampler,
        num_workers=cfg.data.workers,
        pin_memory=True,
        drop_last=True,  # Since moco expects proper batches
    )
    if cfg.apex:
        raise NotImplementedError()

    lr_scheduler = hydra.utils.instantiate(cfg.opt.scheduler,
                                           optimizer,
                                           iters_per_epoch=len(data_loader))

    last_saved_ckpt = 'checkpoint.pth'
    start_epoch = 0
    if os.path.isfile(last_saved_ckpt):
        checkpoint = torch.load(last_saved_ckpt, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        start_epoch = checkpoint['epoch'] + 1
        logger.warning('Loaded model from %s (ep %d)', last_saved_ckpt,
                       start_epoch)

    logger.info("Start training")
    start_time = time.time()

    # Get training metric logger
    stat_loggers = get_default_loggers(writer, start_epoch, logger)

    for epoch in range(start_epoch, cfg.train.num_epochs):
        if dist_info['distributed']:
            train_sampler.set_epoch(epoch)
        train_one_epoch(
            model,
            criterion,
            optimizer,
            lr_scheduler,
            data_loader,
            device,
            epoch,
            cfg.train.print_freq,
            stat_loggers["train"],
            cfg.apex,
        )
        if epoch % cfg.train.save_freq == 0:
            checkpoint = {
                "model": model_without_ddp.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "epoch": epoch,
            }
            utils.save_on_master(checkpoint,
                                 os.path.join('model_{}.pth'.format(epoch)))
            utils.save_on_master(checkpoint, last_saved_ckpt)

        # reset all meters in the metric logger
        for log in stat_loggers:
            stat_loggers[log].reset_meters()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time %d', total_time_str)


def train_one_epoch(
        model,
        criterion,
        optimizer,
        lr_scheduler,
        data_loader,
        device,
        epoch,
        print_freq,
        metric_logger,
        apex=False,
):
    model.train()
    header = "Epoch: [{}]".format(epoch)
    for data in metric_logger.log_every(data_loader, print_freq, header):
        videos, _, orig_target = data
        start_time = time.time()
        videos[0], videos[1], orig_target = (videos[0].to(device),
                                             videos[1].to(device),
                                             orig_target.to(device))
        output, target = model(im_q=videos[0], im_k=videos[1])
        loss = criterion(output, target)

        optimizer.zero_grad()
        if apex:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        optimizer.step()

        acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
        batch_size = videos[0].shape[0]
        metric_logger.update(loss=loss.item(),
                             lr=optimizer.param_groups[0]["lr"])
        metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
        metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)
        metric_logger.meters["clips/s"].update(batch_size /
                                               (time.time() - start_time))
        lr_scheduler.step()
