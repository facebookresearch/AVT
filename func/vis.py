""" Based on https://gist.github.com/zlapp/40126608b01a5732412da38277db9ff5 """
import logging
import os
from scipy.special import softmax
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
from tqdm import tqdm
import moviepy
from moviepy.editor import ImageSequenceClip, TextClip, CompositeVideoClip

import torch

from func.train import DATASET_EVAL_CFG_KEY, RESULTS_SAVE_DIR
import models
from notebooks import utils as nb_utils

activation = {}
NOT_LABELED = '[Not Labeled]'


def _overlay_scale_magnitude(im, mask, peakify=2):
    return ((mask**peakify) * im).astype("uint8")


def _overlay_red(im, mask, peakify=2):
    im = np.array(im)[:, :, ::-1]  # Convert to BGR
    hsvIm = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    # multiply by a factor to change the saturation
    hsvIm[..., 1] = (hsvIm[..., 1] * 0.5).astype(np.uint8)
    im = cv2.cvtColor(hsvIm, cv2.COLOR_HSV2BGR)
    im = im[:, :, ::-1]  # Back to RGB
    im = np.tile(np.mean(im, axis=-1, keepdims=True), (1, 1, 3))
    scale_down = np.ones_like(mask) * 0.8
    red_ch = np.tile(mask**peakify, (1, 1, 3))
    red_ch[:, :, 1:] = 0.0
    red_ch /= red_ch.max()
    red_ch *= 128.0
    # Threshold to make it clearer
    red_ch[red_ch < 60] = 0
    combined = scale_down * im + red_ch
    combined[combined > 255] = 255
    return combined.astype("uint8")


# overlay_fn = _overlay_scale_magnitude
overlay_fn = _overlay_red
PEAKIFY_ATTENTION = 2  # Raise to this power the attention values when visualizing
TXT_FNTSZ_SIDE = 10
TXT_FNTSZ = 15
ATT_LINE_WIDTH = 10


def get_attn_softmax(name):
    def hook(model, input, output):
        with torch.no_grad():
            input = input[0]
            B, N, C = input.shape
            qkv = (model.qkv(input).detach().reshape(
                B, N, 3, model.num_heads,
                C // model.num_heads).permute(2, 0, 3, 1, 4))
            q, k, _ = (
                qkv[0],
                qkv[1],
                qkv[2],
            )  # make torchscript happy (cannot use tensor as tuple)
            attn = (q @ k.transpose(-2, -1)) * model.scale
            attn = attn.softmax(dim=-1)
            activation[name] = attn

    return hook


# expects timm vis transformer model
def add_attn_vis_hook(model):
    for idx, module in enumerate(list(model.blocks.children())):
        module.attn.register_forward_hook(get_attn_softmax(f"attn{idx}"))


def attention_rollout(att_mat):
    """
    Args:
        att_mat: (nlayer, nhead, seqlen, seqlen)
    Returns:
        joint_attention: (nlayer, seqlen, seqlen)
    """
    # Average the attention weights across all heads.
    # att_mat,_ = torch.max(att_mat, dim=1)
    att_mat = torch.mean(att_mat, dim=1)

    # To account for residual connections, we add an identity matrix to the
    # attention matrix and re-normalize the weights.
    residual_att = torch.eye(att_mat.size(1), device=att_mat.device)
    aug_att_mat = att_mat + residual_att
    aug_att_mat = aug_att_mat / aug_att_mat.sum(dim=-1).unsqueeze(-1)

    # Recursively multiply the weight matrices
    joint_attentions = torch.zeros(aug_att_mat.size(),
                                   device=aug_att_mat.device)
    joint_attentions[0] = aug_att_mat[0]

    for n in range(1, aug_att_mat.size(0)):
        joint_attentions[n] = torch.matmul(aug_att_mat[n],
                                           joint_attentions[n - 1])
    return joint_attentions


def attention_vit_overlay(im, att_mat):
    """
    Args:
        im: PIL.Image of H, W, 3 dimension
        att_mat: (nlayer, nhead, 197, 197) where 197 = CLS token + (14*14)
    """
    # Using attention rollout
    joint_attentions = attention_rollout(att_mat)
    # Attention from the output token to the input space.
    v = joint_attentions[-1]
    grid_size = int(np.sqrt(att_mat.size(-1)))
    # 0th element is the CLS token, so taking its attention over the image
    mask = v[0, 1:].reshape(grid_size, grid_size).detach().cpu().numpy()
    mask = cv2.resize(mask / mask.max(), im.size)[..., np.newaxis]
    result = overlay_fn(im, mask)
    return result, joint_attentions, grid_size


def attention_gpt(att_mat):
    """
    Args:
        att (nlayer, nhead, seqlen, seqlen)
        seq_points: The output points to return attention for.
    Returns:
        (seqlen, seqlen) <-- how much i attends to j
    """
    joint_attentions = attention_rollout(att_mat)
    # By the last flow-ed attention, most of the weight on the fist token,
    # since it has the most paths to it (given the causal nature of GPT).
    # So lets just visualize the first layer, which might be more informative
    # final_att = joint_attentions[-1]
    # Even the following seemed to mostly focus on the last frame
    # final_att = joint_attentions[0]
    # Trying average all layers and heads
    # final_att = torch.mean(att_mat, dim=[0, 1])
    # Trying last layer
    final_att = torch.mean(att_mat, dim=1)[-1]
    return final_att


def is_interesting(data, outputs, batch_id):
    last_pred = outputs['past_logits/action'][batch_id][-1].cpu().numpy()
    future_pred = outputs['logits/action'][batch_id].cpu().numpy()
    # Visualize the ones where the model predicts the correct thing within
    # the top 5
    if (data['target']['action'].cpu().numpy()[batch_id]
            not in np.argpartition(future_pred, -5)[-5:]):
        return False
    # The predicted action is different from the last predicted one
    if future_pred.argmax() == last_pred.argmax():
        return False
    # The predicted action is predicted with high conf
    if softmax(future_pred).max() < 0.7:
        return False
    return True


def save_vis_as_graph(outputs, data, uid, im, batch, nframes, im_att,
                      gpt_attention, cls_id_name,
                      frames_to_show_connections_for):
    # Rows
    FRAME_ROW = 2
    ATT_ROW = 1
    PRED_ROW = 0

    ht = 4
    nrows = 3
    # wd / ht
    frame_aspect_ratio = im_att.shape[2] / im_att.shape[1]
    fig, ax = plt.subplots(
        nrows,
        nframes + 1,
        gridspec_kw={'height_ratios': [0.6] + [1] * (nrows - 1)},
        figsize=(ht * (nframes + 2) * frame_aspect_ratio / nrows, ht))
    # All axis off
    for frame_id in range(nframes + 1):
        for row_id in range(nrows):
            ax[row_id, frame_id].axis('off')
    for frame_id in range(nframes):
        # Show the future action, so that will be the past logits
        # output at the next frame, since I shift it by 1
        # (pre-pending the input feature) when returning past
        # logits.
        if frame_id < nframes - 1:
            # + 1 since the past_logits show the current action
            # (including the way we pad the last features as is)
            # while I want show the future one for consistency
            preds = (outputs['past_logits/action'][batch,
                                                   frame_id + 1].cpu().numpy())
            gt = data['target_subclips']['action'].cpu().numpy()[batch,
                                                                 frame_id +
                                                                 1][0]
        elif frame_id == nframes - 1:
            preds = outputs['logits/action'][batch].cpu().numpy()
            gt = data['target']['action'].cpu().numpy()[batch]
        # ax[GT_ROW, frame_id+1].text(
        #     0,
        #     0.5,
        #     cls_id_name[gt].replace(' ', '\n') if gt >= 0 else '',
        #     {'ha': 'left', 'va': 'bottom'},
        #     fontsize=TXT_FNTSZ)
        fontcolor = 'green' if (gt == preds.argmax() or gt == -1) else 'black'
        preds = softmax(preds)
        cls_name = cls_id_name[preds.argmax()]
        kwargs = {}
        if frame_id in frames_to_show_connections_for:
            kwargs['bbox'] = dict(facecolor='none',
                                  edgecolor='grey',
                                  boxstyle='round,pad=0.2')
            # Now, only showing future actions for the selected
            # frames.
            TXT_HORIZONTAL_POS = 0.5
            ax[PRED_ROW, frame_id + 1].text(TXT_HORIZONTAL_POS,
                                            0.9,
                                            cls_name,
                                            ha='center',
                                            va='top',
                                            fontsize=TXT_FNTSZ,
                                            color=fontcolor,
                                            **kwargs)
        ax[ATT_ROW, frame_id].imshow(im_att[frame_id], aspect='auto')
        ax[FRAME_ROW, frame_id].imshow((im[batch][frame_id]), aspect='auto')
    # Draw the lines between the subplots weighted with the att
    for i in frames_to_show_connections_for:
        for j in range(nframes):
            attention_val = gpt_attention[i, j]
            if attention_val > 0.0:
                con = ConnectionPatch(xyA=(TXT_HORIZONTAL_POS, 0.5),
                                      xyB=(224 * TXT_HORIZONTAL_POS, 0),
                                      coordsA='data',
                                      coordsB='data',
                                      axesA=ax[PRED_ROW, i + 1],
                                      axesB=ax[ATT_ROW, j],
                                      color='olive',
                                      linewidth=attention_val * ATT_LINE_WIDTH)
                ax[PRED_ROW, i + 1].add_artist(con)

    plt.subplots_adjust(hspace=0.05, wspace=0.05)
    nb_utils.save_graph(fig, f'{uid}.pdf', root_dir='./', dpi=300)


def save_vis_as_video(outputs, data, uid, im_all, batch, nframes, im_att,
                      cls_id_name):
    output_fps = 32
    stop_duration = 3
    output_width = 768
    fontsize = 24
    total_stopped = 0
    frame_pos = data['video_frames_subsampled'][batch].cpu().numpy()
    im_all = im_all[batch]
    # replace the key frames with the attention
    for frame_id in range(nframes):
        im_all[frame_pos[frame_id]] = im_att[frame_id]
    video = ImageSequenceClip(list(im_all), fps=output_fps)
    video = video.resize(width=output_width)
    txt_clips = []
    for frame_id in range(nframes):
        if frame_id < nframes - 1:
            # + 1 since the past_logits show the current action
            # (including the way we pad the last features as is)
            # while I want show the future one for consistency
            preds = (outputs['past_logits/action'][batch,
                                                   frame_id + 1].cpu().numpy())
            gt = data['target_subclips']['action'].cpu().numpy()[batch,
                                                                 frame_id +
                                                                 1][0]
        elif frame_id == nframes - 1:
            preds = outputs['logits/action'][batch].cpu().numpy()
            gt = data['target']['action'].cpu().numpy()[batch]
        pred_cls = cls_id_name[preds.argmax()]
        gt_cls = cls_id_name[gt] if gt >= 0 else NOT_LABELED
        time_of_overlay = total_stopped + frame_pos[frame_id] / output_fps
        video = moviepy.video.fx.all.freeze(video,
                                            t=time_of_overlay,
                                            freeze_duration=stop_duration)
        # Now overlay the text
        caption = (TextClip(
            f'Predicted future: {pred_cls}\nGT Future: {gt_cls}',
            color='yellow',
            bg_color='black',
            align='West',
            fontsize=fontsize,
            font='DejaVu-Sans-Mono',
        ).set_start(time_of_overlay).set_duration(stop_duration).set_fps(
            output_fps).set_position(('left', 'top')))
        txt_clips.append(caption)
        total_stopped += stop_duration
    video = CompositeVideoClip([video] + txt_clips)
    video.write_videofile(f'{uid}.mp4', threads=10, audio=False)


def visualize(train_eval_op,
              data_loaders: dict,
              tb_writer,
              logger,
              epoch: float,
              only_run_featext: bool = True,
              frames_to_show_connections_for: list = [4, 9],
              uid_subset: list = None):
    """
    Visualize the attention maps.
    """
    del tb_writer, epoch
    assert uid_subset is None, (
        'Please use dataset_eval.uid_subset to define the UID subsets now. '
        'That is a lot more efficient than running through the full dataset '
        'and dropping stuff here.')
    del uid_subset
    global activation
    assert only_run_featext, 'Has to be True'
    for data_key, data_loader in data_loaders.items():
        logger.info('Running visualization for {0}{1}'.format(
            DATASET_EVAL_CFG_KEY, data_key))
        this_save_dir = RESULTS_SAVE_DIR + data_key + '/'
        os.makedirs(this_save_dir, exist_ok=True)
        # only for VIT backbone
        assert isinstance(train_eval_op.model.backbone,
                          models.video_classification.TIMMModel)
        add_attn_vis_hook(train_eval_op.model.backbone.model)
        cls_names = list(data_loader.dataset.classes.values())[0]
        cls_id_name = {v: k for k, v in cls_names.items()}
        for data in tqdm(data_loader, desc='Gen vis'):
            batch_uids = data['uid']
            activation = {}  # clear it out
            with torch.no_grad():
                _, outputs, _, _ = train_eval_op(data, train_mode=False)

            def unprocess(im):
                return ((im * 0.5 + 0.5) * 255).cpu().numpy().astype('uint8')

            assert data['video'].size(2) == 1, 'Run with 1 crops during vis'
            # im will be B x T x H x W x 3
            im = unprocess(data['video'].squeeze(2).squeeze(3).permute(
                0, 1, 3, 4, 2))
            # all frames, including ones that were skipped to get the frames
            # that were processed
            # im_all will be B x T' x H x W x 3
            # permut in this case is different because this tensor does not
            # go through the "subclip" process, which moves the temporal
            # dimension further up
            im_all = unprocess(
                data['video_without_fps_subsample'].squeeze(1).permute(
                    0, 2, 3, 4, 1))
            attn_weights_list = list(activation.values())
            nframes = data['video'].size(1)

            ### PLOT and VIS
            for batch in range(data['video'].shape[0]):
                uid = batch_uids[batch]
                if not is_interesting(data, outputs, batch):
                    # Lets only plot the interesting cases
                    continue
                ### TEMPORAL ATTENTION
                # Use the attention for the original string and visualize
                # For futher rolled out models, need TODO
                assert 'gpt2_att_0' in outputs, (
                    'Make sure +model.future_predictor.output_attentions=true '
                    'is passed in the config.')
                gpt_attention = attention_gpt(outputs['gpt2_att_0'][batch])
                ### SPATIAL ATTENTION
                # Get mask for each element of the batch and time. Looping is
                # just much faster than making the code work with batch dim
                # unprocess the video back to image
                im_att = []
                for frame_id in range(im[batch].shape[0]):
                    idx = batch * nframes + frame_id
                    att, _, _ = attention_vit_overlay(
                        Image.fromarray(im[batch][frame_id]),
                        torch.cat(
                            [el[idx:idx + 1] for el in attn_weights_list]))
                    im_att.append(att)
                    # cv2.imwrite(f'{this_save_dir}/temp_{i:02f}.jpg',
                    #             result[:, :, ::-1])
                im_att = np.stack(im_att).reshape((nframes, ) +
                                                  im_att[0].shape)
                save_vis_as_graph(outputs, data, uid, im, batch, nframes,
                                  im_att, gpt_attention, cls_id_name,
                                  frames_to_show_connections_for)
                save_vis_as_video(outputs, data, uid, im_all, batch, nframes,
                                  im_att, cls_id_name)


def visualize_rollout(train_eval_op,
                      data_loaders: dict,
                      tb_writer,
                      logger,
                      epoch: float,
                      only_run_featext: bool = True,
                      num_futures: int = 2):
    """
    Visualize the long term rollouts.
    """
    # Rows
    FRAME_ROW = 2
    PRED_ROW = 1
    GT_ROW = 0
    NROWS = 3
    HT = 4
    del tb_writer, epoch
    assert only_run_featext, 'Has to be True'
    for data_key, data_loader in data_loaders.items():
        logger.info('Running visualization for {0}{1}'.format(
            DATASET_EVAL_CFG_KEY, data_key))
        this_save_dir = RESULTS_SAVE_DIR + data_key + '/'
        os.makedirs(this_save_dir, exist_ok=True)
        cls_names = list(data_loader.dataset.classes.values())[0]
        cls_id_name = {v: k for k, v in cls_names.items()}
        for data in tqdm(data_loader, desc='Gen vis'):
            with torch.no_grad():
                _, outputs, _, _ = train_eval_op(data, train_mode=False)

            assert data['video'].size(2) == 6, (
                'Run with 3 crops/flips as normal')
            # im will be B x T x H x W x 3
            im = ((data['video'][:, :, 1, ...].squeeze(3).permute(
                0, 1, 3, 4, 2) * 0.5 + 0.5) *
                  255).cpu().numpy().astype('uint8')
            nframes = data['video'].size(1)
            total_cols = nframes + num_futures

            box_kwargs = {}
            box_kwargs['bbox'] = dict(facecolor='none',
                                      edgecolor='grey',
                                      boxstyle='round,pad=0.2')

            ### PLOT and VIS
            for batch in range(data['video'].shape[0]):
                ht = HT
                nrows = NROWS
                top_ratio = 0.15
                # wd / ht
                frame_aspect_ratio = im.shape[-2] / im.shape[-3]
                fig, ax = plt.subplots(
                    nrows,
                    total_cols,
                    gridspec_kw={
                        'height_ratios': [top_ratio] + [1] * (nrows - 1),
                        'width_ratios': [1] * (total_cols - 1) + [0.5],
                    },
                    figsize=((ht * total_cols * frame_aspect_ratio) /
                             (nrows - 1 + top_ratio), ht))
                # All axis off
                for frame_id in range(total_cols):
                    for row_id in range(nrows):
                        ax[row_id, frame_id].axis('off')
                assert outputs['logits/action'].ndim == 3, (
                    'predict longer term future and not avg last N')
                preds = outputs['logits/action'][batch].cpu().numpy()
                preds = softmax(preds)
                # No need of past_logits here since I'm not averaging over the
                # last N so the logits will contain all past predictions as
                # well

                if data['target']['action'].cpu().numpy()[batch] != preds[
                        nframes - 1].argmax():
                    # Only plotting those where the first prediction
                    # with real inputs at least matches the GT
                    continue

                for frame_id in range(nframes):
                    # Show the future action, so that will be the past logits
                    # output at the next frame, since I shift it by 1
                    # (pre-pending the input feature) when returning past
                    # logits.
                    if frame_id < nframes - 1:
                        gt = data['target_subclips']['action'].cpu().numpy()[
                            batch, frame_id + 1][0]
                    elif frame_id == nframes - 1:
                        gt = data['target']['action'].cpu().numpy()[batch]
                    pred = preds[frame_id].argmax()

                    # fontcolor = 'green' if (gt == pred
                    #                         or gt == -1) else 'black'
                    # Just using black to avoid any confusion in this figure
                    # Since can't really do the same for the future actions
                    fontcolor = 'black'
                    cls_name = cls_id_name[pred]
                    TXT_HT = 0.5
                    ax[PRED_ROW, frame_id].text(0.5,
                                                TXT_HT,
                                                cls_name,
                                                ha='center',
                                                va='top',
                                                fontsize=TXT_FNTSZ,
                                                color=fontcolor,
                                                **box_kwargs)
                    ax[FRAME_ROW, frame_id].imshow((im[batch][frame_id]),
                                                   aspect='auto')
                ax[GT_ROW, nframes // 2].text(
                    0.5,
                    TXT_HT,
                    "Ground truth future:",
                    ha='center',
                    va='top',
                    fontsize=TXT_FNTSZ * 1.2,
                    color=fontcolor,
                )
                # Collect the futures into a run length encoding
                preds_future = preds[nframes:]
                preds_future_top1 = preds_future.argmax(axis=-1)
                preds_future_top1_runlen = []
                for p in preds_future_top1:
                    if len(preds_future_top1_runlen
                           ) == 0 or preds_future_top1_runlen[-1][0] != p:
                        preds_future_top1_runlen.append([p, 1])
                    else:
                        preds_future_top1_runlen[-1][1] += 1
                # Collect GT in similar run len
                gt_future_runlen = []
                for p in data['future_subclips']['action'][batch]:
                    if len(gt_future_runlen
                           ) == 0 or gt_future_runlen[-1][0] != p:
                        gt_future_runlen.append([p, 1])
                    else:
                        gt_future_runlen[-1][1] += 1
                for i in range(
                        min(
                            num_futures,
                            max(len(preds_future_top1_runlen),
                                len(gt_future_runlen)))):
                    if i < len(preds_future_top1_runlen):
                        cls_name = cls_id_name[preds_future_top1_runlen[i][0]]
                        cls_cnt = preds_future_top1_runlen[i][1]
                        ax[PRED_ROW,
                           nframes + i].text(0.05,
                                             TXT_HT,
                                             ':'.join([cls_name,
                                                       str(cls_cnt)]),
                                             ha='left',
                                             va='top',
                                             fontsize=TXT_FNTSZ,
                                             color='black',
                                             **box_kwargs)
                    if i < len(gt_future_runlen):
                        gt_cls = gt_future_runlen[i][0].item()
                        gt_cls_name = cls_id_name[
                            gt_cls] if gt_cls >= 0 else NOT_LABELED
                        gt_cls_cnt = gt_future_runlen[i][1]
                        ax[GT_ROW, nframes + i].text(
                            0.05,
                            TXT_HT,
                            ':'.join([gt_cls_name,
                                      str(gt_cls_cnt)]),
                            ha='left',
                            va='top',
                            fontsize=TXT_FNTSZ,
                            color='black',
                            **box_kwargs)
                plt.subplots_adjust(hspace=0.05, wspace=0.05)
                nb_utils.save_graph(fig,
                                    f'{data["uid"][batch]}.pdf',
                                    root_dir='./',
                                    dpi=300)
