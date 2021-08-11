from __future__ import print_function
from typing import List, Dict

import errno
import os
from pathlib import Path
import logging
import submitit
import cv2

import torch
import torch.distributed as dist


def accuracy(output, target, topk=(1, )):
    """Computes the accuracy over the k top predictions
    for the specified values of k
    Args:
        output (*, K) predictions
        target (*, ) targets
    """
    if torch.all(target < 0):
        return [
            torch.zeros([], device=output.device) for _ in range(len(topk))
        ]
    with torch.no_grad():
        # flatten the initial dimensions, to deal with 3D+ input
        output = output.flatten(0, -2)
        target = target.flatten()
        # Now compute the accuracy
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target[None])

        res = []
        for k in topk:
            correct_k = correct[:k].flatten().sum(dtype=torch.float32)
            res.append(correct_k * (100.0 / batch_size))
        return res


def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def setup_for_distributed(is_master, logger):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print
    if not is_master:
        # Don't print anything except FATAL
        logger.setLevel(logging.ERROR)
        logging.basicConfig(level=logging.ERROR)
    else:
        logger.setLevel(logging.INFO)
        logging.basicConfig(level=logging.INFO)


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(logger, dist_backend='nccl'):
    dist_info = dict(
        distributed=False,
        rank=0,
        world_size=1,
        gpu=0,
        dist_backend=dist_backend,
        dist_url=get_init_file(None).as_uri(),
    )
    # If launched using submitit, get the job_env and set using those
    try:
        job_env = submitit.JobEnvironment()
    except RuntimeError:
        job_env = None
    if job_env is not None:
        dist_info['rank'] = job_env.global_rank
        dist_info['world_size'] = job_env.num_tasks
        dist_info['gpu'] = job_env.local_rank
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        dist_info['rank'] = int(os.environ["RANK"])
        dist_info['world_size'] = int(os.environ['WORLD_SIZE'])
        dist_info['gpu'] = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        dist_info['rank'] = int(os.environ['SLURM_PROCID'])
        dist_info['gpu'] = dist_info['rank'] % torch.cuda.device_count()
    elif 'rank' in dist_info:
        pass
    else:
        print('Not using distributed mode')
        dist_info['distributed'] = False
        return dist_info

    dist_info['distributed'] = True

    torch.cuda.set_device(dist_info['gpu'])
    dist_info['dist_backend'] = dist_backend
    print('| distributed init (rank {}): {}'.format(dist_info['rank'],
                                                    dist_info['dist_url']),
          flush=True)
    torch.distributed.init_process_group(backend=dist_info['dist_backend'],
                                         init_method=dist_info['dist_url'],
                                         world_size=dist_info['world_size'],
                                         rank=dist_info['rank'])
    setup_for_distributed(dist_info['rank'] == 0, logger)
    return dist_info


def get_shared_folder(name) -> Path:
    # Since using hydra, which figures the out folder
    return Path('./').absolute()
    # if Path("/checkpoint/").is_dir():
    #     return Path(f"/checkpoint/bkorbar/experiments/{name}")
    # raise RuntimeError("No shared folder available")


def get_init_file(name):
    # Init file must not exist, but it's parent dir must exist.
    os.makedirs(str(get_shared_folder(name)), exist_ok=True)
    # Not using a uuid thing since I want all the workers to get the same URI
    # And since the job runs in a unique folder, can just use that
    init_file = get_shared_folder(name) / 'sync_file_init'
    # # Lets not do the following.. sometimes it seems it deletes the file
    # # while another process is trying to use it? I am anyway clearing
    # # this file in the launch.py
    # if init_file.exists():
    #     try:
    #         os.remove(str(init_file))
    #     except FileNotFoundError:
    #         pass  # Sometimes there is a race condition
    return init_file


def gather_tensors_from_all(tensor: torch.Tensor) -> List[torch.Tensor]:
    """
    Wrapper over torch.distributed.all_gather for performing
    'gather' of 'tensor' over all processes in both distributed /
    non-distributed scenarios.
    """
    if tensor.ndim == 0:
        # 0 dim tensors cannot be gathered. so unsqueeze
        tensor = tensor.unsqueeze(0)

    if is_dist_avail_and_initialized():
        gathered_tensors = [
            torch.zeros_like(tensor)
            for _ in range(torch.distributed.get_world_size())
        ]
        torch.distributed.all_gather(gathered_tensors, tensor)
    else:
        gathered_tensors = [tensor]

    return gathered_tensors


def gather_from_all(tensor: torch.Tensor) -> torch.Tensor:
    gathered_tensors = gather_tensors_from_all(tensor)
    gathered_tensor = torch.cat(gathered_tensors, 0)
    return gathered_tensor


def get_video_info(video_path: Path, props: List[str]) -> Dict[str, float]:
    """
    Given the video, return the properties asked for
    """
    output = {}
    cam = cv2.VideoCapture(str(video_path))
    if 'fps' in props:
        output['fps'] = cam.get(cv2.CAP_PROP_FPS)
    if 'len' in props:
        # Adding a small EPS to avoid crashes
        fps = cam.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            output['len'] = 0
        else:
            output['len'] = (cam.get(cv2.CAP_PROP_FRAME_COUNT) / fps)
    cam.release()
    return output
