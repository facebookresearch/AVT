# Copyright (c) Facebook, Inc. and its affiliates.

"""Utils for notebook."""

import sys
import os
import os.path as osp
import glob
from collections import OrderedDict
from collections.abc import Iterable
import json
import subprocess
import pickle as pkl
import logging
import h5py
import math
import operator
import pathlib

import pandas as pd
import moviepy.editor as mpy
from tqdm import tqdm
import proglog
import numpy as np
from scipy.special import softmax
import torch
# from omegaconf import OmegaConf
import hydra
from hydra.experimental import initialize as hydra_initialize, compose as hydra_compose

import matplotlib
from matplotlib import pylab
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns
# from tqdm import tqdm
from tqdm.notebook import tqdm

sys.path.append('..')
from external.rulstm.RULSTM.utils import topk_recall
from launch import subselect_dict_keys_diff
from datasets import epic_kitchens

CODE_DIR = str(pathlib.Path(__file__).parent.resolve() / '../')
OUTPUT_DIR = f'{CODE_DIR}/OUTPUTS/'
RESULTS_SAVE_DIR_PREFIX = 'results'  # This is the prefix, can have multiple, if >1 eval datasets
DATASET_EVAL_CFG_KEY = 'dataset_eval'
DATASET_EVAL_CFG_KEY_SUFFIX = ''
proglog.notebook()  # so moviepy uses notebook tqdm

SQRT2 = math.sqrt(2)
sns.set_style("whitegrid")
rcParams['mathtext.fontset'] = 'custom'
rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
rcParams['mathtext.fontset'] = 'stix'
rcParams['font.family'] = 'STIXGeneral'
matplotlib.rc('axes', edgecolor='k')
matplotlib.rc('font', size=30)


def save_graph(fig, outfpath, root_dir='./', **kwargs):
    # Any postprocessing of the graphs
    sns.despine(top=True, right=True, left=False, bottom=False)
    # Save code
    final_oufpath = os.path.join(root_dir, outfpath)
    os.makedirs(osp.dirname(final_oufpath), exist_ok=True)
    fig.savefig(final_oufpath,
                bbox_inches='tight',
                transparent=True,
                pad_inches=0,
                **kwargs)


def allkeys(obj, keys=[]):
    """Recursively find all leaf keys in h5. """
    keys = []
    for key in obj.keys():
        if isinstance(obj[key], h5py.Group):
            keys += [f'{key}/{el}' for el in allkeys(obj[key])]
        else:
            keys.append(key)
    return keys


class EmptyResdirError(ValueError):
    pass


def gen_load_resfiles(resdir):
    resfiles = glob.glob(osp.join(resdir, '*.pth'))
    if len(resfiles) == 0:
        resfiles = glob.glob(osp.join(resdir, '*.h5'))
    if len(resfiles) == 0:
        raise EmptyResdirError(f'Didnt find any resfiles in {resdir}')
    for resfile in resfiles:
        if resfile.endswith('.pth'):
            output_dict = {
                key: val.numpy() if torch.torch.is_tensor(val) else val
                for key, val in torch.load(resfile).items()
            }
        else:
            output_dict = {}
            with h5py.File(resfile, 'r') as fin:
                for key in allkeys(fin):
                    try:
                        output_dict[key] = fin[key][()]
                    except AttributeError as err:
                        # Happens for the string keys... need to figure what
                        # to do here
                        logging.warning('Unable to load %s (%s)', key, err)
        yield output_dict


def read_results(conf_path, run_id=0, results_dir='results/'):
    resdir = osp.join(OUTPUT_DIR, conf_path, str(run_id), results_dir)
    data = next(gen_load_resfiles(resdir))
    # TODO allow to read only certain keys, eg some times we only need logits
    # which would be faster to read
    res_per_layer = {
        key: OrderedDict()
        for key in data if key not in ['epoch']
    }
    if len(res_per_layer) == 0:
        raise ValueError('No logits found in the output. Note that code was '
                         'changed Aug 26 2020 that renames "output" to '
                         '"logits" etc. So might need to rerun testing.')
    logging.info('Reading from resfiles')
    for data in gen_load_resfiles(resdir):
        for i, idx in enumerate(data['idx']):
            idx = int(idx)
            for key in res_per_layer:
                if idx not in res_per_layer[key]:
                    res_per_layer[key][idx] = []
                res_per_layer[key][idx].append(data[key][i])
    # Mean over all the multiple predictions per key
    final_res = {}
    for key in res_per_layer:
        if len(res_per_layer[key]) == 0:
            continue
        max_idx = max(res_per_layer[key].keys())
        key_output = np.zeros([
            max_idx + 1,
        ] + list(res_per_layer[key][0][0].shape))
        for idx in res_per_layer[key]:
            key_output[idx] = np.mean(np.stack(res_per_layer[key][idx]),
                                      axis=0)
        final_res[key] = key_output
    return final_res


def get_epoch_from_resdir(conf_path, run_id=0, results_dir='results/'):
    resdir = osp.join(OUTPUT_DIR, conf_path, str(run_id), results_dir)
    data = next(gen_load_resfiles(resdir))
    if 'epoch' not in data:
        return None
    return np.min(data['epoch'])


def read_all_results(conf_path, run_id=0):
    resdirs = glob.glob(
        osp.join(OUTPUT_DIR, conf_path, str(run_id),
                 RESULTS_SAVE_DIR_PREFIX + '*'))
    all_res = {}
    for resdir in resdirs:
        resdir_bname = osp.basename(resdir)
        all_res[resdir_bname] = read_results(conf_path,
                                             run_id,
                                             results_dir=resdir_bname)
    return all_res


def read_file_into_list(fpath):
    """Read cli from file into a string."""
    # TODO: Ideally reuse this from the launch script
    args_lst = []
    with open(fpath, 'r') as fin:
        for line in fin:
            args = line.split('#')[0].strip()
            if not args:  # Empty
                continue
            args_lst.append(args)
    # Importing this on the global scope does not work .. gives the
    # super(cls, self).. error
    # https://thomas-cokelaer.info/blog/2011/09/382/
    # Probably some issue with auto package reload in notebooks for py2.7
    # packages..
    from hydra._internal.core_plugins.basic_sweeper import BasicSweeper
    from hydra.core.override_parser.overrides_parser import OverridesParser
    sweeper = BasicSweeper(max_batch_size=None)
    parser = OverridesParser.create()
    overrides = parser.parse_overrides(args_lst)
    run_args = sweeper.split_arguments(overrides, max_batch_size=None)[0]
    return run_args


def get_config(cfg_fpath, run_id=0):
    # outdir = osp.join(OUTPUT_DIR, cfg_fpath, str(run_id))
    overrides_all = read_file_into_list('../' + cfg_fpath)
    # https://github.com/facebookresearch/hydra/issues/716 should fix the issue
    # with interpolation not working in notebook etc.
    # However it can't handle ":" style custom interpolation, so need to
    # override those.
    cfg_all = []
    for overrides in overrides_all:
        overrides.append('cwd="../"')
        with hydra_initialize(config_path='../conf'):
            cfg = hydra_compose(config_name='config.yaml',
                                return_hydra_config=True,
                                overrides=overrides)
        cfg_all.append(cfg)
    if run_id is None:
        return cfg_all
    else:
        return cfg_all[run_id]


def get_dataset(cfg_fpath,
                run_id=0,
                dataset_cfg_key=DATASET_EVAL_CFG_KEY,
                dataset_key_suffix=DATASET_EVAL_CFG_KEY_SUFFIX):
    cfg = get_config(cfg_fpath, run_id)
    sys.path.append('../')
    dataset = hydra.utils.instantiate(getattr(
        cfg, dataset_cfg_key + dataset_key_suffix),
                                      frames_per_clip=1,
                                      _recursive_=False)
    return dataset


def overlay_text(clip, texts):
    """
    Args:
        clip: Moviepy clip
        texts: List of 2 strings (corr to GT and pred) to overlay onto the clip
    """
    bg_color = 'white' if texts[0] == texts[1] else 'pink'
    texts[0] = 'GT: ' + texts[0]
    texts[1] = 'Pred: ' + texts[1]
    textclip = (mpy.TextClip(str(texts), bg_color=bg_color).set_duration(
        clip.duration).set_pos(("right", "top")))
    return mpy.CompositeVideoClip([clip, textclip])


def compute_topk(predictions, labels, k, classes=None):
    """
    Args:
        predictions (N, K)
        labels (N,)
        classes: (C', ): Set of classes to compute over. By default, uses
            all classes
    """
    if classes is None:
        classes = np.unique(labels)
    # Subselect items that belong to the classes
    # Converting to list since classses are at times dict_values and that
    # doesn't directly convert to np.array
    reqd_elts = np.isin(labels, list(classes))
    predictions = predictions[reqd_elts]
    labels = labels[reqd_elts]
    top_predictions = np.argpartition(predictions, -k, axis=-1)[:, -k:]
    ratio_solved = np.mean(
        np.any(labels[:, np.newaxis] == top_predictions, axis=-1))
    return ratio_solved * 100.0


def combine_verb_noun_preds(res_verb, res_noun):
    """
    Args:
        res_verb (matrix with NxC1 dims)
        res_noun (matrix with NxC2 dims)
    Returns:
        res_action (matrix with Nx(C1 * C2) dims)
    """
    num_elts = res_verb.shape[0]
    # normalize the predictions using softmax
    res_verb = softmax(res_verb, axis=-1)
    res_noun = softmax(res_noun, axis=-1)
    # Cross product to get the combined score
    return np.einsum('ij,ik->ijk', res_verb, res_noun).reshape((num_elts, -1))


def compute_conf_mat(predictions, target):
    def to_onehot(indices, num_classes):
        onehot = torch.zeros(indices.shape[0],
                             num_classes,
                             *indices.shape[1:],
                             device=indices.device)
        # rgirdhar: When test on test set, there will be some data points where
        # we don't have the labels
        return onehot.scatter_(1, indices[indices >= 0].unsqueeze(1), 1)

    num_classes = predictions.shape[1]
    assert predictions.shape[0] == target.shape[0]
    with torch.no_grad():
        target_1hot = to_onehot(target, num_classes)
        target_1hot_t = target_1hot.transpose(0, 1).float()

        pred_idx = torch.argmax(predictions, dim=1)
        pred_1hot = to_onehot(pred_idx.reshape(-1), num_classes)
        pred_1hot = pred_1hot.float()

        confusion_matrix = torch.matmul(target_1hot_t, pred_1hot)
    return confusion_matrix


def mean_class_accuracy(conf_mat):
    # Increase floating point precision similar to forecasting HOI
    conf_mat = conf_mat.type(torch.float64)
    cls_cnt = conf_mat.sum(dim=1) + 1e-15
    cls_hit = conf_mat.diag()
    cls_acc = (cls_hit / cls_cnt).mean().item()
    return cls_acc


def compute_accuracy(predictions, labels, classes=None):
    """
    Args:
        predictions: (B, C) logits
        labels: (B, )
        classes: OrderedDict[name (str), cls_id (int)]
    """
    # This can happen when computing tail class accuracies and it's not
    # specified for the test set
    if predictions.size == 0:
        return [float('nan')] * 5
    labels = labels.astype(np.int64)
    if classes is not None:
        classes_to_keep = list(classes.values())
    else:
        classes_to_keep = range(max(labels) + 1)
    top_1 = compute_topk(predictions, labels, 1, classes=classes_to_keep)
    top_5 = compute_topk(predictions, labels, 5, classes=classes_to_keep)
    try:
        ar_outputs = topk_recall(predictions,
                                 labels,
                                 k=5,
                                 classes=classes_to_keep)
        if isinstance(ar_outputs, tuple):
            # This happens if RULSTM code is modified to return per-class AR
            # values
            ar5, ar5_per_cls = ar_outputs
            ar5_per_cls = {k: v * 100.0 for k, v in ar5_per_cls.items()}
        else:
            ar5 = ar_outputs
            ar5_per_cls = {c: float('nan') for c in classes_to_keep}
    except ZeroDivisionError:
        # This happens when it can't find any true classes, the code
        # can't deal with that
        ar5 = float('nan')
        ar5_per_cls = {c: float('nan') for c in classes_to_keep}
    # Compute a mean class accuracy (used in EGTEA) -- accuracy per class and
    # then mean over the classes
    conf_mat = compute_conf_mat(torch.from_numpy(predictions),
                                torch.from_numpy(labels))
    # Make sure conf mat makes sense
    top_1_confmat = 100.0 * (conf_mat.diag()[classes_to_keep].sum() /
                             conf_mat[classes_to_keep].sum())
    if (not np.isnan(top_1) and not np.isnan(top_1_confmat)
            and not np.isclose(top_1, top_1_confmat, atol=1.0)):
        # Using a large atol margin cos conf_mat comp happens on GPUs and can
        # be non deterministic, so might not match sometimes..
        # Save the outputs for analysis
        with open('debug_acc.pkl', 'wb') as fout:
            pkl.dump(predictions, fout)
            pkl.dump(labels, fout)
            pkl.dump(conf_mat, fout)
        raise ValueError(f'top1 ({top_1}) doesnt match what I get from '
                         f'conf_mat ({top_1_confmat}). This could happen '
                         f'if the model predicts all 0s for some data points '
                         f'and hence argmax is not defined and behaves '
                         f'differently in numpy and torch '
                         f'(https://github.com/pytorch/pytorch/issues/14147)')
    top1_meancls = 100.0 * mean_class_accuracy(conf_mat)
    return top_1, top_5, ar5 * 100, top1_meancls, ar5_per_cls


def print_accuracies_epic(metrics: dict, prefix: str = ''):
    print(f"[{prefix}] Accuracies verb/noun/action: "
          f"{metrics['vtop1']:.1f} {metrics['vtop5']:.1f} "
          f"{metrics['ntop1']:.1f} {metrics['ntop5']:.1f} "
          f"{metrics['atop1']:.1f} {metrics['atop5']:.1f} ")
    print(f"[{prefix}] Mean class top-1 accuracies verb/noun/action: "
          f"{metrics['vtop1_meancls']:.1f} "
          f"{metrics['ntop1_meancls']:.1f} "
          f"{metrics['atop1_meancls']:.1f} ")
    print(f"[{prefix}] Recall@5 verb/noun/action: "
          f"{metrics['vrec5']:.1f} {metrics['nrec5']:.1f} "
          f"{metrics['arec5']:.1f} ")
    print(f"[{prefix}] Recall@5 many shot verb/noun/action: "
          f"{metrics['vrec5_ms']:.1f} {metrics['nrec5_ms']:.1f} "
          f"{metrics['arec5_ms']:.1f} ")
    if 'vrec5_tail' in metrics:
        # assuming the others for tail/unseen will be in there too, since
        # they are all computed at one place for ek100
        print(f"[{prefix}] Recall@5 tail verb/noun/action: "
              f"{metrics['vrec5_tail']:.1f} {metrics['nrec5_tail']:.1f} "
              f"{metrics['arec5_tail']:.1f} ")
        print(f"[{prefix}] Recall@5 unseen verb/noun/action: "
              f"{metrics['vrec5_unseen']:.1f} {metrics['nrec5_unseen']:.1f} "
              f"{metrics['arec5_unseen']:.1f} ")


def get_logits_from_results(results):
    if 'logits' in results:
        return results['logits']
    # Newer version, as of Nov 3 2020
    logits_keys = [key for key in results.keys() if key.startswith('logits/')]
    if len(logits_keys) == 1:
        return results[logits_keys[0]]
    # Else, return all of them in a dict
    return {key: results[key] for key in logits_keys}


def get_epic_action_accuracy(run_info_verb, run_info_noun):
    # Compute action accuracies implicitly from verb and noun
    # TODO also compute with many-shot classes for EPIC 55
    res_verb = get_logits_from_results(read_results(*run_info_verb))
    res_noun = get_logits_from_results(read_results(*run_info_noun))
    dataset_verb = get_dataset(*run_info_verb)
    vtop1, vtop5, vrec5, vtop1_meancls, vrec5_per_cls = compute_accuracy(
        res_verb, dataset_verb.df['verb_class'].values)
    dataset_noun = get_dataset(*run_info_noun)
    ntop1, ntop5, nrec5, ntop1_meancls, nrec5_per_cls = compute_accuracy(
        res_noun, dataset_noun.df['noun_class'].values)
    assert (len(dataset_verb.df) == len(res_verb) == len(dataset_noun.df) ==
            len(res_noun))
    res_action = combine_verb_noun_preds(res_verb, res_noun)
    true_action = (
        dataset_verb.df['verb_class'].values * len(dataset_noun.classes) +
        dataset_noun.df['noun_class'].values)
    atop1, atop5, arec5, atop1_meancls, arec5_per_cls = compute_accuracy(
        res_action, true_action)
    print_accuracies_epic({
        'vtop1': vtop1,
        'vtop5': vtop5,
        'vrec5': vrec5,
        'vrec5_ms': float('nan'),  # TODO
        'vtop1_meancls': vtop1_meancls,
        'vrec5_per_cls': vrec5_per_cls,
        'ntop1': ntop1,
        'ntop5': ntop5,
        'nrec5': nrec5,
        'nrec5_ms': float('nan'),  # TODO
        'ntop1_meancls': ntop1_meancls,
        'nrec5_per_cls': nrec5_per_cls,
        'atop1': atop1,
        'atop5': atop5,
        'arec5': arec5,
        'arec5_ms': float('nan'),  # TODO
        'atop1_meancls': atop1_meancls,
        'arec5_per_cls': arec5_per_cls,
    })


def epic100_unseen_tail_eval(probs, dataset):
    """
    probs: contains 3 elements: predictions for verb, noun and action
    """
    # based on https://github.com/fpv-iplab/rulstm/blob/d44612e4c351ff668f149e2f9bc870f1e000f113/RULSTM/main.py#L379
    unseen_participants_ids = pd.read_csv(osp.join(
        dataset.rulstm_annotation_dir,
        'validation_unseen_participants_ids.csv'),
                                          names=['id'],
                                          squeeze=True)
    tail_verbs_ids = pd.read_csv(osp.join(dataset.rulstm_annotation_dir,
                                          'validation_tail_verbs_ids.csv'),
                                 names=['id'],
                                 squeeze=True)
    tail_nouns_ids = pd.read_csv(osp.join(dataset.rulstm_annotation_dir,
                                          'validation_tail_nouns_ids.csv'),
                                 names=['id'],
                                 squeeze=True)
    tail_actions_ids = pd.read_csv(osp.join(dataset.rulstm_annotation_dir,
                                            'validation_tail_actions_ids.csv'),
                                   names=['id'],
                                   squeeze=True)
    # Now based on https://github.com/fpv-iplab/rulstm/blob/d44612e4c351ff668f149e2f9bc870f1e000f113/RULSTM/main.py#L495
    unseen_bool_idx = dataset.df.narration_id.isin(
        unseen_participants_ids).values
    tail_verbs_bool_idx = dataset.df.narration_id.isin(tail_verbs_ids).values
    tail_nouns_bool_idx = dataset.df.narration_id.isin(tail_nouns_ids).values
    tail_actions_bool_idx = dataset.df.narration_id.isin(
        tail_actions_ids).values
    # For tail
    _, _, vrec5_tail, _, _ = compute_accuracy(
        probs[0][tail_verbs_bool_idx],
        dataset.df.verb_class.values[tail_verbs_bool_idx])
    _, _, nrec5_tail, _, _ = compute_accuracy(
        probs[1][tail_nouns_bool_idx],
        dataset.df.noun_class.values[tail_nouns_bool_idx])
    _, _, arec5_tail, _, _ = compute_accuracy(
        probs[2][tail_actions_bool_idx],
        dataset.df.action_class.values[tail_actions_bool_idx])
    # for unseen
    _, _, vrec5_unseen, _, _ = compute_accuracy(
        probs[0][unseen_bool_idx],
        dataset.df.verb_class.values[unseen_bool_idx])
    _, _, nrec5_unseen, _, _ = compute_accuracy(
        probs[1][unseen_bool_idx],
        dataset.df.noun_class.values[unseen_bool_idx])
    _, _, arec5_unseen, _, _ = compute_accuracy(
        probs[2][unseen_bool_idx],
        dataset.df.action_class.values[unseen_bool_idx])
    return dict(
        vrec5_tail=vrec5_tail,
        nrec5_tail=nrec5_tail,
        arec5_tail=arec5_tail,
        vrec5_unseen=vrec5_unseen,
        nrec5_unseen=nrec5_unseen,
        arec5_unseen=arec5_unseen,
    )


def compute_accuracies_epic(probs, dataset):
    manyshot_classes = dataset.classes_manyshot
    vtop1, vtop5, vrec5, vtop1_meancls, vrec5_per_cls = compute_accuracy(
        probs[0], dataset.df.verb_class.values)
    vrec5_ms, nrec5_ms, arec5_ms = float('nan'), float('nan'), float('nan')
    if 'verb' in manyshot_classes:
        _, _, vrec5_ms, _, _ = compute_accuracy(
            probs[0],
            dataset.df.verb_class.values,
            classes=manyshot_classes['verb'])
    ntop1, ntop5, nrec5, ntop1_meancls, nrec5_per_cls = compute_accuracy(
        probs[1], dataset.df.noun_class.values)
    if 'noun' in manyshot_classes:
        _, _, nrec5_ms, _, _ = compute_accuracy(
            probs[1],
            dataset.df.noun_class.values,
            classes=manyshot_classes['noun'])
    atop1, atop5, arec5, atop1_meancls, arec5_per_cls = compute_accuracy(
        probs[2], dataset.df.action_class.values)
    if 'action' in manyshot_classes:
        _, _, arec5_ms, _, _ = compute_accuracy(
            probs[2],
            dataset.df.action_class.values,
            classes=manyshot_classes['action'])
    res = {
        'vtop1': vtop1,
        'vtop5': vtop5,
        'vrec5': vrec5,
        'vrec5_ms': vrec5_ms,
        'vtop1_meancls': vtop1_meancls,
        'vrec5_per_cls': vrec5_per_cls,
        'ntop1': ntop1,
        'ntop5': ntop5,
        'nrec5': nrec5,
        'nrec5_ms': nrec5_ms,
        'ntop1_meancls': ntop1_meancls,
        'nrec5_per_cls': nrec5_per_cls,
        'atop1': atop1,
        'atop5': atop5,
        'arec5': arec5,
        'arec5_ms': arec5_ms,
        'atop1_meancls': atop1_meancls,
        'arec5_per_cls': arec5_per_cls,
    }
    if dataset.version == epic_kitchens.EPIC100_VERSION:
        res.update(epic100_unseen_tail_eval(probs, dataset))
    return res


def get_epic_marginalize_verb_noun(
        run_info, dataset_key_suffix=DATASET_EVAL_CFG_KEY_SUFFIX):
    res_action = get_logits_from_results(
        read_results(*run_info, results_dir=f'results{dataset_key_suffix}'))
    dataset = get_dataset(*run_info, dataset_key_suffix=dataset_key_suffix)
    if isinstance(res_action, dict):
        print(f'Found logits outputs for verb noun as well [{run_info}]')
        # It has multiple heads for verb/noun as well
        res_verb = res_action['logits/verb']
        res_noun = res_action['logits/noun']
        res_action = res_action['logits/action']
    else:
        res_action_probs = softmax(res_action, axis=-1)
        # Marginalize the other dimension, using the mapping matrices I store
        # in the dataset obj
        res_verb = np.matmul(
            res_action_probs,
            dataset.class_mappings[('verb', 'action')].numpy())
        res_noun = np.matmul(
            res_action_probs,
            dataset.class_mappings[('noun', 'action')].numpy())
    accuracies = compute_accuracies_epic([res_verb, res_noun, res_action],
                                         dataset)
    # Returning the actual scores for actions instead of the probs. Found
    # better results with this, and Sener et al. ECCV'20 does the same.
    scores = [res_verb, res_noun, res_action]
    return accuracies, scores, dataset


def read_scores_from_pkl(pkl_fpath):
    """
    This is to read the data as I dump in the ActionBanks code
    """
    with open(pkl_fpath, 'rb') as fin:
        scores = pkl.load(fin)
    return [
        scores['verb_scores'], scores['noun_scores'], scores['action_scores']
    ]


def load_json(fpath, verb_noun_to_action, nclasses):
    """
    Args:
        fpath: Path to the json
        verb_noun_to_action: Dict from (verb_id, noun_id) to action_id
        nclasses: A list of 3 elements, with the label space for verb/noun/act
    Returns: a dict with
    {uid1: score1, uid2: score2 ...}
    """
    assert len(nclasses) == 3, 'One for verb/noun/action'
    with open(fpath, 'r') as fin:
        preds = json.load(fin)
    # Res for verb/noun/action
    all_res = []
    for j, space in enumerate(['verb', 'noun', 'action']):
        # Convert to a {uid: <scores>} format
        res = {}
        for key, val in preds['results'].items():
            # Will be using 0 for all the scores not defined. Should be fine given
            # top 100 should be enough for late fusion etc, metrics are like top-5
            # anyway.
            scores = np.zeros((nclasses[j], ))
            for i, score in val[space].items():
                if space == 'action':
                    # Since for actions the "key" is (verb, noun) tuple,
                    # need to convert it to an action index by
                    # verb_id * noun_count + noun_id
                    idx = tuple(int(el) for el in i.split(','))
                    idx = verb_noun_to_action[idx]
                else:
                    idx = int(i)
                scores[idx] = score
            res[key] = scores
        all_res.append(res)
    return all_res


def _concat_with_uids(scores, dataset, uid_key):
    # Make a dict with the IDs from the dataset
    # There will be 3 elements in scores -- verb, noun, action
    return [
        dict(
            zip([str(el)
                 for el in dataset.df[uid_key].values], scores_per_space))
        for scores_per_space in scores
    ]


def _normalize_scores(scores, p):
    """This brings the scores between 0 to 1, and normalizes by """
    res = []
    for scores_per_space in scores:
        res.append({
            uid: val / (np.linalg.norm(val, ord=p, axis=-1) + 0.000001)
            for uid, val in scores_per_space.items()
        })
    return res


def _get_avg_norm_scores(scores, p):
    """Remove the UID keys etc, and then compute."""
    scores = np.array([val for _, val in scores.items()])
    return np.mean(np.linalg.norm(scores, ord=p, axis=-1), axis=0)


def get_epic_marginalize_late_fuse(
        run_infos,
        weights=1.0,
        dataset_key_suffix=DATASET_EVAL_CFG_KEY_SUFFIX,
        uid_key='uid',
        eventual_fname='seen.json',
        normalize_before_combine=None):
    """
    Args:
        eventual_fname: This is used to read prepackaged outputs from result
            files, and using the filename to know which file to look for
            when a directory is passed in as run info.
        normalize_before_combine: Set to non-None to normalize the features
            by that p-norm, and then combine. So the weights would have to be
            defined w.r.t normalized features.
    """
    all_scores = []
    all_datasets = []
    for run_info_id, run_info in enumerate(run_infos):
        if isinstance(run_info[0], dict):
            # This is likely a pre-computed scores (so eg a nested
            # get_epic_marginalize.. function). So I just use the scores as is.
            scores = run_info
        elif os.path.isdir(run_info[0]):
            assert len(all_datasets) > 0, (
                'Need at least 1 datasets to be read before reading from json '
                'to figure the verb/noun -> action_id and '
                'to figure the total number of classes to gen feat vectors')
            scores = load_json(
                os.path.join(run_info[0], eventual_fname),
                all_datasets[-1].verb_noun_to_action,
                [list(el.values())[0].shape[-1] for el in all_scores[-1]])
        elif run_info[0].endswith('.pkl'):
            # This is the input used to read predictions from the action_banks
            # codebase, where I dump output into pkl and read here for late
            # fusion.
            scores = read_scores_from_pkl(run_info[0])
            assert len(
                all_datasets) > 0, 'At least one run_info must be passed in'
            scores = _concat_with_uids(scores, all_datasets[-1], uid_key)
        else:
            accuracies, scores, dataset = get_epic_marginalize_verb_noun(
                run_info, dataset_key_suffix=dataset_key_suffix)
            scores = _concat_with_uids(scores, dataset, uid_key)
            print_accuracies_epic(accuracies, prefix=run_info)
            all_datasets.append(dataset)
        if normalize_before_combine is not None:
            scores = _normalize_scores(scores, p=normalize_before_combine)
        logging.warning(
            'Adding scores from run_info %d with avg action L1 norm of %f',
            run_info_id, _get_avg_norm_scores(scores[-1], p=1))
        all_scores.append(scores)
    # Late fuse
    if isinstance(weights, float):
        weights = [weights] * len(run_infos)
    else:
        assert len(weights) == len(run_infos)
    # broadcastable_weights = np.array(weights)[:, np.newaxis, np.newaxis]
    # Combined scores by combining the corresponding score for each uid.
    combined = []
    for space_id in range(3):  # verb/noun/action
        scores_for_space = [scores[space_id] for scores in all_scores]
        # Take the union of all the UIDs we have score for
        total_uids = set.union(*[set(el.keys()) for el in scores_for_space])
        logging.warning('Combined UIDs: %d. UIDs in the runs %s',
                        len(total_uids),
                        [len(el.keys()) for el in scores_for_space])
        combined_for_space = {}
        for uid in total_uids:
            combined_for_space[uid] = []
            for run_id, scores_for_space_per_run in enumerate(
                    scores_for_space):
                if uid in scores_for_space_per_run:
                    combined_for_space[uid].append(
                        scores_for_space_per_run[uid] * weights[run_id])
            combined_for_space[uid] = np.sum(np.stack(combined_for_space[uid]),
                                             axis=0)
        combined.append(combined_for_space)
    # Now to compute accuracies, need to convert back to np arrays from dict.
    # Would only work for parts that are in the dataset
    combined_np = []
    for combined_for_space in combined:
        combined_np.append(
            np.array([
                combined_for_space[str(uid)]
                for uid in all_datasets[-1].df[uid_key].values
            ]))
    accuracies = compute_accuracies_epic(combined_np, all_datasets[-1])
    return accuracies, combined, all_datasets[-1]


def summarize_results(cfg_name, metric='arec5'):
    """
    Read all runs corr to cfg_name, and show the results in a human readable
    form with the config overrides (unique) that were active. It averages
    over runs too.
    """
    run_cfgs = read_file_into_list('../' + cfg_name)
    run_cfgs_hydra = get_config(cfg_name, run_id=None)
    # Convert to dicts
    run_cfgs = [(i, dict([el.split('=') for el in conf]))
                for i, conf in enumerate(run_cfgs)]
    # Keep only the stuff that changes across them
    run_cfgs = subselect_dict_keys_diff(run_cfgs)
    all_res = {}
    for (run_id, params), cfg_hydra in tqdm(zip(run_cfgs, run_cfgs_hydra),
                                            total=len(run_cfgs),
                                            desc='Loading results'):
        try:
            accuracies, _, _ = get_epic_marginalize_verb_noun(
                (cfg_name, run_id))
            epoch = get_epoch_from_resdir(cfg_name, run_id)
        except (EmptyResdirError, OSError):  # H5 didn't let it read
            continue
        if epoch != cfg_hydra.train.num_epochs:
            # This training has not finished
            continue
        run_id = 0
        if 'run_id' in params:
            run_id = int(params['run_id'])
            del params['run_id']
        params_hash = tuple(sorted(list(params.items())))
        if params_hash not in all_res:
            all_res[params_hash] = {}
        all_res[params_hash][run_id] = accuracies[metric]
    for params_hash in all_res:
        run_ids, values = zip(*all_res[params_hash].items())
        print(f'{params_hash} [{run_ids}]: [{values}] '
              f'mean: {np.mean(values)}, std: {np.std(values)}')


def plot_per_cls_perf(run_infos_all: list,
                      names: list,
                      metrics: list = ['vrec5_per_cls', 'nrec5_per_cls'],
                      cls_types: list = ['verb', 'noun'],
                      show_topn: int = 10,
                      xticks_rotation: float = 0,
                      show_subset: callable = None,
                      outfpath: str = 'figs/improved/'):
    """
    Args:
        run_infos_all: [[(cfg, sweep_id), (cfg, sweep_id)...],
                        [(cfg, sweep_id), (cfg, sweep_id)...], ...]
        names: The name for each run_info group
        metrics: There will be 1 graph for each
    """
    assert len(run_infos_all) == len(names)
    assert len(metrics) == len(cls_types)
    final_accs = {cls_type: [] for cls_type in cls_types}
    for i, run_infos in enumerate(tqdm(run_infos_all, desc='Reading acc')):
        for run_id, run_info in enumerate(run_infos):
            cfg_fpath, sweep_id = run_info
            all_accuracies, _, dataset = get_epic_marginalize_verb_noun(
                (cfg_fpath, sweep_id))
            for metric, cls_type in zip(metrics, cls_types):
                accuracies = all_accuracies[metric]
                assert isinstance(accuracies,
                                  dict), 'Supports per-class for now'
                classes = operator.attrgetter(f'{cls_type}_classes')(dataset)
                cls_id_to_name = {v: k for k, v in classes.items()}
                for cls_id, score in accuracies.items():
                    final_accs[cls_type].append({
                        'method':
                        names[i],
                        'run_id':
                        run_id,
                        'cls_name':
                        cls_id_to_name[cls_id],
                        'accuracy':
                        score,
                    })
    for cls_type in final_accs:
        accs = pd.DataFrame(final_accs[cls_type])
        # Print logs
        for method in names:
            for run_id in accs.run_id.unique():
                this_acc = (accs[accs.method == method][
                    accs.run_id == run_id].accuracy.mean())
                print(f'Check {method} {run_id}: {this_acc}')
        mean_acc_by_cls = accs.groupby(['method',
                                        'cls_name']).mean().reset_index()
        first_col = mean_acc_by_cls[mean_acc_by_cls.method == names[0]]
        last_col = mean_acc_by_cls[mean_acc_by_cls.method == names[-1]]
        merged = first_col[['cls_name', 'accuracy'
                            ]].merge(last_col[['cls_name', 'accuracy']],
                                     on='cls_name',
                                     how='outer',
                                     suffixes=['_first', '_last'])
        # get the largest gains
        gains = (merged['accuracy_last'] -
                 merged['accuracy_first']).sort_values()
        gained_labels = merged.loc[gains.index].cls_name.tolist()
        if show_subset is not None:
            gained_labels = [el for el in gained_labels if show_subset(el)]
        gained_labels = gained_labels[-show_topn:]
        accs_largegains = accs[accs.cls_name.isin(gained_labels)]
        fig = plt.figure(num=None,
                         figsize=(2 * len(gained_labels), 4),
                         dpi=300)
        ax = sns.barplot(x='cls_name',
                         y='accuracy',
                         hue='method',
                         data=accs_largegains,
                         order=gained_labels,
                         errwidth=1.0)
        ax.set_xlabel('Classes')
        ax.set_ylabel('Recall @ 5')
        ax.set_xticklabels(ax.get_xticklabels(),
                           rotation=xticks_rotation,
                           ha='center')
        plt.show()
        save_graph(fig, os.path.join(outfpath, cls_type + '.pdf'))


def get_struct_outputs_per_dataset(run_infos,
                                   weights,
                                   dataset_key_suffix,
                                   uid_key='uid',
                                   eventual_fname='seen.json',
                                   normalize_before_combine=None):
    _, combined, dataset = get_epic_marginalize_late_fuse(
        run_infos,
        weights,
        dataset_key_suffix=dataset_key_suffix,
        uid_key=uid_key,
        eventual_fname=eventual_fname,
        normalize_before_combine=normalize_before_combine)
    results = {}
    # Now the following may not be true since if the run_info contains an
    # actual json, it might have more rows etc.
    # assert len(combined[0]) == len(dataset)
    action_to_verb_noun = {
        val: key
        for key, val in dataset.verb_noun_to_action.items()
    }
    for uid in tqdm(combined[0].keys(), desc='Computing res'):
        verb_res = {f'{j}': val for j, val in enumerate(combined[0][uid])}
        noun_res = {f'{j}': val for j, val in enumerate(combined[1][uid])}
        top_100_actions = sorted(np.argpartition(combined[2][uid],
                                                 -100)[-100:],
                                 key=lambda x: -combined[2][uid][x])
        action_res = {
            ','.join((str(el)
                      for el in action_to_verb_noun[j])): combined[2][uid][j]
            for j in top_100_actions
        }
        results[f'{uid}'] = {
            'verb': verb_res,
            'noun': noun_res,
            'action': action_res,
        }
    # Add in all the discarded dfs with uniform distribution
    if dataset.discarded_df is not None:
        for _, row in dataset.discarded_df.iterrows():
            if str(row[uid_key]) in results:
                continue
            results[f'{row[uid_key]}'] = {
                'verb':
                {f'{j}': 0.0
                 for j in range(len(dataset.verb_classes))},
                'noun':
                {f'{j}': 0.0
                 for j in range(len(dataset.noun_classes))},
                'action': {f'0,{j}': 0.0
                           for j in range(100)},
            }
    output_dict = {
        'version': f'{dataset.version}',
        'challenge': dataset.challenge_type,
        'results': results
    }
    return output_dict


def package_results_for_submission(run_infos,
                                   weights,
                                   normalize_before_combine=None):
    res_s1 = get_struct_outputs_per_dataset(
        run_infos,
        weights,
        dataset_key_suffix='',
        eventual_fname='seen.json',
        normalize_before_combine=normalize_before_combine)
    res_s2 = get_struct_outputs_per_dataset(
        run_infos,
        weights,
        dataset_key_suffix='_s2',
        eventual_fname='unseen.json',
        normalize_before_combine=normalize_before_combine)
    # write it out in the first run's output dir
    output_dir = osp.join(OUTPUT_DIR, run_infos[0][0], str(run_infos[0][1]),
                          'challenge')
    print(f'Saving outputs to {output_dir}')
    os.makedirs(output_dir, exist_ok=True)
    with open(osp.join(output_dir, 'seen.json'), 'w') as fout:
        json.dump(res_s1, fout, indent=4)
    with open(osp.join(output_dir, 'unseen.json'), 'w') as fout:
        json.dump(res_s2, fout, indent=4)
    subprocess.check_output(
        f'zip -j {output_dir}/submit.zip '
        f'{output_dir}/seen.json '
        f'{output_dir}/unseen.json ',
        shell=True)


def package_results_for_submission_ek100(run_infos, weights, sls=[1, 4, 4]):
    res = get_struct_outputs_per_dataset(run_infos,
                                         weights,
                                         dataset_key_suffix='',
                                         uid_key='narration_id',
                                         eventual_fname='test.json')
    res['sls_pt'] = sls[0]
    res['sls_tl'] = sls[1]
    res['sls_td'] = sls[2]
    # write it out in the first run's output dir
    output_dir = osp.join(OUTPUT_DIR, run_infos[0][0], str(run_infos[0][1]),
                          'challenge')
    print(f'Saving outputs to {output_dir}')
    os.makedirs(output_dir, exist_ok=True)
    with open(osp.join(output_dir, 'test.json'), 'w') as fout:
        json.dump(res, fout, indent=4)
    subprocess.check_output(
        f'zip -j {output_dir}/submit.zip '
        f'{output_dir}/test.json ',
        shell=True)
