"""
Taken from pytorch image classfication code
This part will probably need refactoring
"""

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

# python imports
import argparse
import os
import time
import math
from pprint import pprint
import json
import numpy as np
import cv2
# torch imports
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.multiprocessing as mp
import torch.nn.functional as F

# our code
from libs.datasets import create_video_dataset_joint, create_video_transforms_joint
from libs.core import load_config
from libs.models import EncoderDecoder as ModelBuilder
from libs.utils import (AverageMeter, accuracy, mean_class_accuracy, confusion_matrix,
                        create_scheduler, fast_clip_collate_joint_test, ClipPrefetcherJointTest)

# the arg parser
parser = argparse.ArgumentParser(
  description='Evaluate 3D ConvNet model on given dataset')
parser.add_argument('config', metavar='DIR',
                    help='path to a config file')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    help='print frequency (default: 10 iterations)')
parser.add_argument('--slice', action='store_true',
                    help='Slice clips from a single video (used for large models)')

# class KLDiv(nn.Module):
#   """
#     KL divergence for 3D attention maps
#   """
#   def __init__(self):
#     super(KLDiv, self).__init__()
#     self.register_buffer('norm_scalar', torch.tensor(1, dtype=torch.float32))

def KLD(pred, target):
  # get output shape
  # print(pred.shape[0])
  batch_size= pred.shape[0]
  H, W = pred.shape[1], pred.shape[2]
  # print(torch.sum(pred))
  # print(torch.sum(target))
  # N T HW
  atten_map = pred.view(batch_size, -1)
  log_atten_map = torch.log(atten_map)


  log_q = torch.log(target.view(batch_size, -1))
  # \sum p logp - \sum p logq -> N T
  kl_losses = (atten_map * log_atten_map).sum(dim=-1) \
              - (atten_map * log_q).sum(dim=-1)
  # N T -> N
  norm_scalar = torch.log(torch.tensor(1, dtype=torch.float32)*H * W)
  kl_losses = kl_losses.sum(dim=-1) / norm_scalar
  kl_loss = kl_losses.mean()
  # print(kl_loss)
  return kl_loss


def compare_hotspot(gt_gaze_map, pred_gaze_map, all_thresh):
  """ compare two gaze maps """
  # return tp, tn, fp, fn

  # expected shape N * T * H * W
  # print(gt_gaze_map.shape)
  # kl = KLDIV()
  for i in range(3):
    assert gt_gaze_map.shape[i] == pred_gaze_map.shape[i]
  # assert gt_gaze_map.shape[-1] == 1



  N = gt_gaze_map.shape[0]
  H, W = gt_gaze_map.shape[1], gt_gaze_map.shape[2]

  # resized_gt_map[]

  resized_gt_map = np.zeros((3,8,8))
  resized_pred_map = np.zeros((3,8,8))
  for i in range(3):
    resized_gt_map[i,:,:] = cv2.resize(gt_gaze_map[i,:,:],(8,8),interpolation = cv2.INTER_AREA)
    # resized_gt_map[i,:,:] = resized_gt_map[i,:,:]/np.sum(resized_gt_map[i,:,:])
    # resized_pred_map[i,:,:] = cv2.resize(pred_gaze_map[i,:,:],(8,8),interpolation = cv2.INTER_AREA)
    resized_pred_map[i,3,3]=1.0 
    blur = cv2.GaussianBlur(resized_pred_map[i,:,:],(3,3),0)
    resized_pred_map[i,:,:] =blur+1e-6
    # resized_pred_map[i,:,:] = resized_pred_map[i,:,:]/np.sum(resized_pred_map[i,:,:])
  
  # resized_gt_map = resized_gt_map/np.max(resized_gt_map)
  resized_pred_map = resized_pred_map/np.max(resized_pred_map)
  # print(np.max(resized_gt_map))
  # print(np.min(resized_gt_map))
  # print(np.max(resized_pred_map))
  # print(np.min(resized_pred_map))
  # print(resized_gt_map.shape)
  # print(resized_pred_map.shape)
  # reshape both gaze maps
  # gt_gaze_map = np.reshape(np.squeeze(gt_gaze_map), (N*T, H, W))
  # pred_gaze_map = np.reshape(np.squeeze(pred_gaze_map), (N*T, H, W))

  # not technicallly correct, yet should be very similar to the true PR
  tp = np.zeros((all_thresh.shape[0],))
  fp = np.zeros((all_thresh.shape[0],))
  fn = np.zeros((all_thresh.shape[0],))
  tn = np.zeros((all_thresh.shape[0],))
  # cropped_ == Trues
  # get valid gaze slices
  valid_slice = []
  cropped = True
  for slice_idx in range(N):
    if np.max(gt_gaze_map[slice_idx, :, :]) > 0.01:
      valid_slice.append(slice_idx)
      cropped = False

  # print(valid_slice)
  # if len(valid_slice) == 0:
  #   print('cropped hotspot map')
  # reslice the data
  valid_gt_gaze = resized_gt_map[valid_slice, :, :]
  valid_gt_gaze = (valid_gt_gaze>0.001)
  valid_pred_gaze = resized_pred_map[valid_slice, :, :]

  for idx, thresh in enumerate(all_thresh):
    mask = (valid_pred_gaze>=thresh)

    tp[idx] += np.sum(np.logical_and(mask==1, valid_gt_gaze==1))
    tn[idx] += np.sum(np.logical_and(mask==0, valid_gt_gaze==0))
    fp[idx] += np.sum(np.logical_and(mask==1, valid_gt_gaze==0))
    fn[idx] += np.sum(np.logical_and(mask==0, valid_gt_gaze==1))



  resized_gt_map = np.zeros((3,8,8))
  resized_pred_map = np.zeros((3,8,8))
  for i in range(3):
    resized_gt_map[i,:,:] = cv2.resize(gt_gaze_map[i,:,:],(8,8),interpolation = cv2.INTER_AREA)
    resized_gt_map[i,:,:] = resized_gt_map[i,:,:]/np.sum(resized_gt_map[i,:,:])
    # resized_pred_map[i,:,:] = cv2.resize(pred_gaze_map[i,:,:],(8,8),interpolation = cv2.INTER_AREA)
    resized_pred_map[i,3,3]=1.0 
    blur = cv2.GaussianBlur(resized_pred_map[i,:,:],(3,3),0)
    resized_pred_map[i,:,:] =blur+1e-6
  # resized_pred_map[i,:,:] = resized_pred_map[i,:,:]/np.sum(resized_pred_map[i,:,:])
  
  pred_tensor = torch.from_numpy(resized_pred_map)
  gt_tensor = torch.from_numpy(resized_gt_map)
  out = torch.nn.KLDivLoss(size_average=True)(pred_tensor.log(),gt_tensor)*64
  # print(out)
  return tp, tn, fp, fn, out, cropped


def compare_hand(hand_gt, hand_pred):

  N,T = hand_gt.shape[0],hand_gt.shape[1]
  H, W = hand_gt.shape[2], hand_gt.shape[3]
  hand_gt = np.reshape(np.squeeze(hand_gt), (N*T, H, W))
  hand_pred = np.reshape(np.squeeze(hand_pred), (N*T, H, W))

  # resized_gt_map = np.zeros((N*T,32,8))
  # resized_pred_map = np.zeros((N*T,8,8))
  # for i in range(N*T):
  #   resized_gt_map[i,:,:] = cv2.resize(hand_gt[i,:,:],(8,8),interpolation = cv2.INTER_AREA)
  #   # resized_gt_map[i,:,:] = resized_gt_map[i,:,:]/np.sum(resized_gt_map[i,:,:])
  #   resized_pred_map[i,:,:] = cv2.resize(hand_pred[i,:,:],(8,8),interpolation = cv2.INTER_AREA)

  # hand_gt = resized_gt_map
  # hand_pred = resized_pred_map

  empty = True
  mean_error = 0.0
  final_error = 0.0
  valid_slice = []
  for slice_idx in range(N*T):
    if np.max(hand_gt[slice_idx, :, :]) > 0.01:
      valid_slice.append(slice_idx)
      empty = False
  if empty == False:
    valid_gt = hand_gt[valid_slice, :, :]
    valid_pred = hand_pred[valid_slice, :, :]

    count = 0
    dist= 0.0
    for j in range(valid_gt.shape[0]):
      gt = valid_gt[j,:,:]
      pred = valid_pred[j,:,:]
      gt_x,gt_y=np.unravel_index(gt.argmax(), gt.shape)
      pred_x,pred_y=np.unravel_index(pred.argmax(), pred.shape)
      dist+=np.sqrt((gt_x-pred_x)**2+(gt_y-pred_y)**2)

    gt = valid_gt[-1,:,:]
    pred = valid_pred[-1,:,:]    
    gt_x,gt_y=np.unravel_index(gt.argmax(), gt.shape)
    pred_x,pred_y=np.unravel_index(pred.argmax(), pred.shape)

    final_error=np.sqrt((gt_x-pred_x)**2+(gt_y-pred_y)**2)
    mean_error = dist/valid_gt.shape[0]

  return mean_error, final_error, empty
# main function for testing
def main(args):
  # parse args
  if os.path.exists(args.config):
    config = load_config(args.config)
  else:
    raise ValueError("Config file does not exist.")
  print("Current configurations:")
  pprint(config)

  # use spawn for mp, this will fix a deadlock by OpenCV
  if mp.get_start_method(allow_none=True) is None:
    mp.set_start_method('spawn')

  # set up transforms and dataset
  _, test_transforms, _ = create_video_transforms_joint(config['input'])
  _, test_dataset = create_video_dataset_joint(config['dataset'], None, test_transforms)
  print("Testing time data augmentations:")
  pprint(test_transforms)

  # skip weight loading if resume from a checkpoint
  if args.resume:
    config['network']['pretrained'] = None
  else:
    print("No model specified. Existing ... ")
    return

  # create model, optimizer and loss function (on GPUs)
  model = ModelBuilder(config['network'])

  # freeze the model
  model.eval()
  for param in model.parameters():
    param.requires_grad = False

  # model -> gpu
  model = model.cuda()
  model = nn.DataParallel(model, device_ids=config['network']['devices'])

  # must resume from a previous checkpoint
  if os.path.isfile(args.resume):
    print("=> loading checkpoint '{}'".format(args.resume))
    checkpoint = torch.load(args.resume)
    best_acc1 = checkpoint['best_acc1']
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    print("=> loaded checkpoint '{}' (epoch {}, acc1 {})"
        .format(args.resume, checkpoint['epoch'], best_acc1))
  else:
    print("=> no checkpoint found at '{}'".format(args.resume))
    return

  # only instantiate the dataset if necessary
  test_dataset = test_dataset()
  test_dataset.load()
  # quick hack: reset the number of lips
  test_dataset.reset_num_clips(config['input']['test_clips'])

  # test batch_size = 1
  test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=len(config['network']['devices']),
    num_workers=config['input']['num_workers'],
    collate_fn=fast_clip_collate_joint_test,
    shuffle=False, pin_memory=True, sampler=None, drop_last=False)

  # evaluation
  model_arch = "{:s}-{:s}".format(
    config['network']['backbone'], config['network']['decoder'])
  print("Testing model {:s}...".format(model_arch))

  # testing: make sure cudnn runs in deterministic mode
  cudnn.enabled = True
  cudnn.benchmark = False
  cudnn.deterministic = True

  # evaluate the model
  validate(test_loader, model, args, config)

  # exit
  print("All done!")


def validate(test_loader, model, args, config):
  """Test the model on the validation set
  We follow "fully convolutional" testing:
    * Scale the video with shortest side =256
    * Uniformly sample 10 clips within a video
    * For each clip, crop K=3 regions of 256*256 along the longest side
    * This is equivalent to 30-crop testing
  """
  # set up meters

  batch_time = AverageMeter()
  top1 = AverageMeter()
  top5 = AverageMeter()
  cm_meter = AverageMeter()

  # data prefetcher with noramlization
  test_loader = ClipPrefetcherJointTest(test_loader,
                               config['input']['mean'],
                               config['input']['std'])

  # loop over validation set
  end = time.time()
  input, target, clipID = test_loader.next()
  i = 0
  mean = config['input']['mean']
  std = config['input']['std']
  mean_tensor = torch.tensor([255.0*val for val in mean]).cuda().view(1,3,1,1,1)
  std_tensor = torch.tensor([255.0*val for val in std]).cuda().view(1,3,1,1,1)
  # for large models
  if args.slice:
    batch_size = input.size(1)
    max_split_size = 1
    for split_size in range(2, batch_size):
      if (batch_size % split_size) == 0 and split_size > max_split_size:
        max_split_size = split_size
    num_batch_splits = batch_size // max_split_size
    print("Split the input by size: {:d}x{:d}".format(
      max_split_size, num_batch_splits))
  

  clipID_list = []
  hand_gt_list = []
  hotspot_gt_list = []
  hand_pred_list = []
  hotspot_pred_list = []
  label_list = []
  score_list = []
  all_thresh = np.linspace(0.001, 1.0, 41)
  tp = np.zeros((all_thresh.shape[0],))
  fp = np.zeros((all_thresh.shape[0],))
  fn = np.zeros((all_thresh.shape[0],))
  tn = np.zeros((all_thresh.shape[0],))
  count=0
  KL=0
  mean_error= 0
  final_error=0

  while input is not None:
    i += 1

    # disable/enable gradients
    with torch.no_grad():
      if args.slice:
        # slice the inputs for testing
        splited_inputs = torch.split(input, max_split_size, dim=1)
        splited_outputs = []
        for idx in range(num_batch_splits):
          split_output = model(splited_inputs[idx])
          # test time augmentation (minor performance boost)
          flipped_split_input = torch.flip(splited_inputs[idx], (-1,))
          flipped_split_output = model(flipped_split_input)
          split_output = 0.5 * (split_output + flipped_split_output)
          splited_outputs.append(split_output)
        output = torch.mean(torch.stack(splited_outputs), dim=0)
      else:
        # forward all inputs
        # print(input.size())
        output,hand_map,hotspot_map = model(input)
        
        # test time augmentation (minor performance boost)
        # always flip the last dim (width)
        flipped_input = torch.flip(input, (-1,))
        flipped_output,_,_ = model(flipped_input)
        output = 0.5 * (output + flipped_output)
      input= input.mul(std_tensor).add_(mean_tensor)
      instance_id = clipID[0].split('.')[0].split('_')[0]
      print('./egtea_vis/'+instance_id+'.jpg')

      # hand_map0 = np.squeeze(hand_map[0,:,:,:].cpu().numpy()) 
      # hand_map1 = np.squeeze(hand_map[1,:,:,:].cpu().numpy()) 
      # hand_map2 = np.squeeze(hand_map[2,:,:,:].cpu().numpy()) 
      hand_map = np.squeeze(hand_map.cpu().numpy()) 
      hotspot_map = np.squeeze(hotspot_map.cpu().numpy()) 
      im = np.squeeze(input[0,:,-1,:,:].cpu().numpy())
      im = im.transpose((1,2,0))
      im = im[..., ::-1]
      # print(hand_map0.shape)
  
      # hand_map_i = hand_map[i,:,:,:]

      # cv2.imwrite('./egtea_vis/'+instance_id+'.jpg',im)
      hotspot_map_i = hotspot_map
      hotspot_map_i = cv2.resize(hotspot_map_i, (256, 256), interpolation=cv2.INTER_LINEAR)
      hotspot_map_i = hotspot_map_i/np.max(hotspot_map_i)*0.7
      hotspot_map_i = (hotspot_map_i*255).astype(np.uint8)
      hotspot_map_i = cv2.applyColorMap(hotspot_map_i, cv2.COLORMAP_JET)
      im = im.astype(np.uint8)
      hotspot_map_i = hotspot_map_i.astype(np.uint8)
      res = cv2.addWeighted(im, 0.5, hotspot_map_i, 0.5, 0)
      cv2.imwrite('./egtea_vis/'+instance_id+'hot.jpg',hotspot_map_i)
      # break
      # assert np.max(res)<=255
      
      # motor_atten=np.zeros((hand_map.shape[1],hand_map.shape[2]))
      # for j in range(hand_map.shape[0]):
      #   motor_atten = motor_atten+hand_map[j,:,:]

      # motor_atten = cv2.resize(motor_atten, (224, 224), interpolation=cv2.INTER_LINEAR)
      # motor_atten = motor_atten/np.max(motor_atten)*0.9
      # motor_atten = (motor_atten*255).astype(np.uint8)
      # motor_atten = cv2.applyColorMap(motor_atten, cv2.COLORMAP_JET)

      # res = im * 0.6 + motor_atten * 0.4      
      # cv2.imwrite('./handvis/'+instance_id+'motoratten.jpg',res)

      # for j in range(hand_map.shape[0]):
      #   t = hand_map[j,:,:]
      #   t = cv2.resize(t, (224, 224), interpolation=cv2.INTER_LINEAR)
      #   pred_y,pred_x=np.unravel_index(t.argmax(), t.shape)
      #   if j == 0:
      #     im = cv2.circle(im, (pred_x,pred_y), 8, (0, 255, 255), -1)
      #   elif j==1:
      #     im = cv2.circle(im, (pred_x,pred_y), 8, (0, 255, 0), -1)
      #   elif j ==2:
      #     im = cv2.circle(im, (pred_x,pred_y), 8, (255, 255, 0), -1)
      #   else:
      #     im = cv2.circle(im, (pred_x,pred_y), 8, (255, 0, 255), -1)
      # cv2.imwrite('./egtea_vis/'+instance_id+'hand.jpg',im)
    # break

      # label_list.append(target[0].cpu().numpy()[0])
      # score_list.append(output.cpu().numpy())
      # hand_gt = np.squeeze(target[1].cpu().numpy())
      # # hotspot_gt = np.squeeze(target[2].cpu().numpy())

      # # # print(hotspot_map)

      # # hand_map = np.squeeze(hand_map.cpu().numpy())

      # hand_map = np.squeeze(hand_map.cpu().numpy())
      # hotspot_map = np.squeeze(hotspot_map.cpu().numpy())
      # # break
      # mean_error_i,final_error_i,empty = compare_hand(hand_gt, hand_map)

      # # ctp, ctn, cfp, cfn,out,cropped = compare_hotspot(hotspot_gt, hotspot_map, all_thresh)
      # # tp = tp + ctp
      # # tn = tn + ctn
      # # fp = fp + cfp
      # # fn = fn + cfn
      # if empty == False:
      #   count+=1
      #   # print(mean_error_i,final_error_i)
      #   mean_error+=mean_error_i
      #   final_error+=final_error_i


    # print(tp,tn,fp,fn)

      # print(output.cpu().numpy().shape)
    # print(np.squeeze(hand_map.cpu().numpy()).shape)
    # print(np.squeeze(target[1].cpu().numpy()).shape)
    # print(np.squeeze(hotspot_map.cpu().numpy()).shape)
    # print(np.squeeze(target[2].cpu().numpy()).shape)
    # hand_pred_list.append(np.squeeze(hand_map.cpu().numpy()).tolist())
    # hotspot_pred_list.append(np.squeeze(hotspot_map.cpu().numpy()).tolist())
    # hand_gt_list.append(np.squeeze(target[1].cpu().numpy()).tolist())
    # hotspot_gt_list.append(np.squeeze(target[2].cpu().numpy()).tolist())     
    # print(output.size())
    # print(target[0].size())
    # print(target[1].size())
    # print(target[2].size())
    # print(clipID[0])
    # print(hand_map.size())
    # print(hotspot_map.size())

    # clipID_list.append(clipID[0])
    # # output_list.append(np.squeeze(output.cpu().numpy()).tolist())

    # # measure accuracy and record loss
    # acc1, acc5 = accuracy(output.data, target[0], topk=(1, 5))
    # top1.update(acc1.item(), input.size(0))
    # top5.update(acc5.item(), input.size(0))
    # batch_cm = confusion_matrix(output.data, target[0])
    # cm_meter.update(batch_cm.data.cpu().double())

    # # # prefetch next batch

    # # # measure elapsed time
    # batch_time.update(time.time() - end)
    # end = time.time()

    # # printing
    # # if i % (args.print_freq * 2) == 0:
    # #   print('Test: [{0}/{1}]\t'
    # #     'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'.format(
    # #      i, len(test_loader), batch_time=batch_time))
    

    # if i % (args.print_freq * 2) == 0:
    #   print('Test: [{0}/{1}]\t'
    #     'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
    #     'Acc@1 {top1.val:.2f} ({top1.avg:.2f})\t'
    #     'Acc@5 {top5.val:.2f} ({top5.avg:.2f})'.format(
    #      i, len(test_loader), batch_time=batch_time,
    #      top1=top1, top5=top5))
    input, target, clipID = test_loader.next()
    # if i >1:
    #   break
    
  # prec = tp / (tp+fp+1e-6)
  # recall = tp / (tp+fn+1e-6)

  # # fix the bad points (make the curve looks nice)
  # prec[-1] = 1.0
  # recall[-1] = 0.0

  # f1 = 2*prec*recall / (prec + recall + 1e-6)
  # idx = np.argmax(f1)
  # print("Gaze Estimation F1 Score {:0.4f} (P={:0.4f}, R={:0.4f}) at th={:0.4f}".format(
  #   f1[idx], prec[idx], recall[idx], all_thresh[idx]))

  # print(count)
  # print(mean_error/count)
  # print(final_error/count)
  # clip_scores = np.vstack(score_list)
  # clip_labels = np.asarray(label_list)
  # print(clip_scores.shape)
  # print(clip_labels)
  # print(len(hand_pred_list))
  # print(len(output_list))
  # print(config['dataset']['ant_file']['val'])
  # action_type = config['dataset']['action_type']
  # name = action_type+'-'+config['dataset']['ant_file']['val'].split('/')[-1]
  # print(name)
  # output_json_file = os.path.join('./test_result/', name)
  # with open(output_json_file, 'w') as fobj:
  #   json.dump({'clipsID': clipID_list,
  #             'hand_gt_list': hand_gt_list,
  #             'hotspot_gt_list': hotspot_gt_list,
  #             'hand_pred_list': hand_pred_list,
  #             'hotspot_pred_list': hotspot_pred_list}, fobj)

  # cls_acc = mean_class_accuracy(cm_meter.sum)
  # print('***Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f} Mean Cls Acc {cls_acc:.3f}'
  #       .format(top1=top1, top5=top5, cls_acc=100*cls_acc))

  # return top1.avg, top5.avg


################################################################################
if __name__ == '__main__':
  args = parser.parse_args()
  main(args)
