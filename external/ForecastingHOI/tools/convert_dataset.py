from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import time
import json
import os
import sys
sys.path.insert(0, os.getcwd())

from functools import partial
from joblib import delayed
from joblib import Parallel

from libs.utils import probe_video
from libs.utils import (load_ant_hmdb51, load_ant_ucf101,
                        load_ant_20bn, load_ant_egtea)

def probe_video_wrapper(video_item, video_folder):
  """A wrapper to probe_video"""
  video_file = os.path.join(video_folder, video_item['filename'])
  video_info, status, log = probe_video(video_file)
  return tuple([video_item['filename'], video_info, status, log])

def process_video_list(video_folder, video_list, num_jobs):
  """Parallel processing of video_list (add video_info)"""
  # put video probe in parallel
  probe_video_func = partial(probe_video_wrapper, video_folder=video_folder)
  start_time = time.time()
  if num_jobs == 1:
    result_list = []
    for video_item in video_list:
      result_list.append(probe_video_func(video_item))
  else:
      result_list = Parallel(n_jobs=num_jobs, verbose=2)(
        delayed(probe_video_func)(video_item) for video_item in video_list)
  end_time = time.time()

  # print the timing
  num_videos = len(video_list)
  print("Processed {:d} videos in {:0.1f} secs".format(
          num_videos, end_time - start_time))
  print("Merging results")

  # attach back the meta info
  status_list = []
  for idx in range(num_videos):
    # unpack all results
    filename, video_info, status, log = result_list[idx]
    # for log file
    status_list.append(tuple([filename, status, log]))
    # for json file
    assert (video_list[idx]['filename'] == filename), "File name mis-match!"
    video_list[idx]['video_info'] = video_info

  return video_list, status_list

def save_results(video_list, label_dict, status_list, dataset_id, output_dir):
  """Save results and progress report"""
  # save report
  output_report_file = './logs/{:s}_report.json'.format(dataset_id)
  with open(output_report_file, 'w') as fobj:
    json.dump(status_list, fobj, indent=2)
  # save video_list / label_dict
  output_json_file = os.path.join(
    output_dir, '{:s}.json'.format(dataset_id))
  with open(output_json_file, 'w') as fobj:
    json.dump({'label_dict': label_dict,
               'video_list': video_list}, fobj)
  print("{:s} has been processed!".format(dataset_id))
  return

def main(video_dir, split_dir, output_dir, dataset, num_jobs=1):

  # create the output folder if possible
  if not os.path.exists(output_dir):
    os.mkdir(output_dir)

  # ucf / hmdb
  if dataset == "ucf101" or dataset == "hmdb51":
    # 3 split / 2 sets
    for split in ['split1', 'split2', 'split3']:
      for set in ['train', 'test']:
        # get split file name
        split_file = os.path.join(split_dir,
          '{:s}_{:s}_{:s}.txt'.format(dataset, split, set))

        # parse annotations
        if dataset == "ucf101":
          video_list, label_dict = load_ant_ucf101(split_file)
        else:
          video_list, label_dict = load_ant_hmdb51(split_file)

        dataset_id = '{:s}_{:s}_{:s}'.format(dataset, split, set)
        print("Start probing videos for {:s}".format(dataset_id))

        # process list
        video_list, status_list = process_video_list(
          video_dir, video_list, num_jobs)

        # save results
        save_results(video_list, label_dict, status_list, dataset_id, output_dir)

  # 20bn-v2
  elif dataset == "20bn-v2":
    # use exisiting label_file
    label_file = os.path.join(split_dir,
                              'something-something-v2-labels.json')
    for set in ['train', 'validation', 'test']:
      json_file = os.path.join(split_dir,
                               'something-something-v2-{:s}.json'.format(set))
      # parse annotations
      video_list, label_dict = \
        load_ant_20bn(json_file, label_file, is_test=(set=='test'))
      dataset_id = '{:s}_{:s}'.format(dataset, set)
      print("Start probing videos for {:s}".format(dataset_id))

      # process list
      video_list, status_list = process_video_list(
        video_dir, video_list, num_jobs)

      # save results
      save_results(video_list, label_dict, status_list, dataset_id, output_dir)

  # EGTEA
  elif dataset == "egtea":
    # action label file
    label_file = os.path.join(split_dir, 'action_idx.txt')
    # 3 split / 2 sets
    for split in ['split1', 'split2', 'split3']:
      for set in ['train', 'test']:
        # load annotations
        split_file = os.path.join(split_dir, '{:s}_{:s}.txt'.format(set, split))
        video_list, label_dict = load_ant_egtea(split_file, label_file)
        dataset_id = '{:s}_{:s}_{:s}'.format(dataset, split, set)
        print("Start probing videos for {:s}".format(dataset_id))

        # process list
        video_list, status_list = process_video_list(
          video_dir, video_list, num_jobs)

        # save results
        save_results(video_list, label_dict, status_list, dataset_id, output_dir)

  else:
    return NotImplemented

if __name__ == '__main__':
  description = 'Helper script for converting dataset annotations.'
  p = argparse.ArgumentParser(description=description)
  p.add_argument('video_dir', type=str,
                 help='Root folder with all video files.')
  p.add_argument('split_dir', type=str,
                 help='Root folder with all split files.')
  p.add_argument('output_dir', type=str,
                 help='Output directory where converted json file will be saved.')
  p.add_argument('-d', '--dataset', type=str, default='ucf101',
                 help='Dataset (ucf101 | hmdb51 | 20bn-v2 | egtea | ...)')
  p.add_argument('-n', '--num-jobs', type=int, default=1)
  main(**vars(p.parse_args()))
