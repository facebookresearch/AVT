# Copyright (c) Facebook, Inc. and its affiliates.

from typing import Sequence

import torch
from bisect import bisect_right


class WarmupMultiStepLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
            self,
            optimizer: torch.optim.Optimizer,
            milestone_epochs: Sequence[int],
            gamma: float = 0.1,
            warmup_factor: float = 1.0 / 3,
            warmup_epochs: int = 5,
            warmup_method: str = 'linear',
            last_epoch: int = -1,
            iters_per_epoch: int = None,  # Must be set by calling code
            world_size: int = None,
    ):
        del world_size
        if not milestone_epochs == sorted(milestone_epochs):
            raise ValueError(
                "Milestones should be a list of"
                " increasing integers. Got {}",
                milestone_epochs,
            )

        if warmup_method not in ("constant", "linear"):
            raise ValueError(
                "Only 'constant' or 'linear' warmup_method accepted"
                "got {}".format(warmup_method))
        self.milestones = [iters_per_epoch * m for m in milestone_epochs]
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_iters = max(warmup_epochs * iters_per_epoch, 1)

        self.warmup_method = warmup_method
        super(WarmupMultiStepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        warmup_factor = 1
        if self.last_epoch < self.warmup_iters:
            if self.warmup_method == "constant":
                warmup_factor = self.warmup_factor
            elif self.warmup_method == "linear":
                alpha = float(self.last_epoch) / self.warmup_iters
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha
        return [
            base_lr * warmup_factor *
            self.gamma**bisect_right(self.milestones, self.last_epoch)
            for base_lr in self.base_lrs
        ]


class CosineLR(torch.optim.lr_scheduler.CosineAnnealingLR):
    def __init__(self,
                 optimizer,
                 num_epochs,
                 iters_per_epoch=None,
                 world_size=None,
                 **kwargs):
        kwargs['eta_min'] *= world_size
        super().__init__(optimizer,
                         T_max=num_epochs * iters_per_epoch,
                         **kwargs)

    def get_lr(self, *args, **kwargs):
        if self.last_epoch < self.T_max:
            return super().get_lr(*args, **kwargs)
        else:
            # Adding this if I train the model longer than the T_max set in
            # this. Happens when I sweep over different amounts of warmup.
            return [0.0 for _ in self.optimizer.param_groups]


class ReduceLROnPlateau(torch.optim.lr_scheduler.ReduceLROnPlateau):
    def __init__(self,
                 optimizer,
                 iters_per_epoch=None,
                 world_size=None,
                 **kwargs):
        del iters_per_epoch, world_size
        super().__init__(optimizer, **kwargs)


class Warmup(torch.optim.lr_scheduler._LRScheduler):
    """Wrap the scheduler for warmup before it kicks in."""
    def __init__(
            self,
            optimizer: torch.optim.Optimizer,
            scheduler: torch.optim.lr_scheduler._LRScheduler,
            init_lr_ratio: float = 0.0,
            num_epochs: int = 5,
            last_epoch: int = -1,
            iters_per_epoch: int = None,  # Must be set by calling code
            world_size: int = None,
    ):
        """
        Args:
            init_lr_ratio (float in [0, 1]): Ratio of the original LR to start
                from. If 0.1, it will start from 0.1 of the original LRs and go
                upto 1.0 of the original LRs in the epochs. By def start from
                0 up.
            num_epochs (int): Num of epochs to take to warmup.
            last_epoch (int): Which was the last epoch to init from (not really
                used anymore since we store the state_dict when loading
                scheduler from disk.)
        """
        del world_size
        self.base_scheduler = scheduler
        self.warmup_iters = max(num_epochs * iters_per_epoch, 1)
        if self.warmup_iters > 1:
            self.init_lr_ratio = init_lr_ratio
        else:
            self.init_lr_ratio = 1.0  # Don't go from 0 to 1 in 1 iteration
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        # Epoch is iters for me, since I step after each iteration
        # (not after each epoch)
        # Based on logic in step, this should only be called for the warmup
        # iters. After that it should go to the base scheduler
        assert self.last_epoch < self.warmup_iters  # since it increments
        return [
            el * (self.init_lr_ratio + (1 - self.init_lr_ratio) *
                  (float(self.last_epoch) / self.warmup_iters))
            for el in self.base_lrs
        ]

    def step(self, *args, **kwargs):
        if self.last_epoch < (self.warmup_iters - 1):
            super().step(*args, **kwargs)
        else:
            self.base_scheduler.step(*args, **kwargs)

    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.

        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """
        base_sched_dict = self.base_scheduler.state_dict()
        other_stuff = {
            key: value
            for key, value in self.__dict__.items() if key not in [
                'base_scheduler', 'optimizer']
        }
        return {'base_sched_dict': base_sched_dict, 'other_stuff': other_stuff}

    def load_state_dict(self, state_dict):
        """Loads the schedulers state.

        Arguments:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.base_scheduler.__dict__.update(state_dict['base_sched_dict'])
        self.__dict__.update(state_dict['other_stuff'])
