"""A set of annotation parsers for different datasets"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json

def load_ant_egtea(csv_file_name, label_file_name, num_classes=106):
  # prep for outputs
  label_dict = {}
  video_list = []

  # load label index
  with open(label_file_name) as f:
    lines = [line.rstrip('\n') for line in f]

  for line in lines:
    tokens = line.split(' ')
    # json serialization won't take integer as keys
    label_id = str(int(tokens[-1]) - 1)
    label_text = ' '.join(tokens[:-1])
    label_dict[label_id] = label_text
  assert len(label_dict) == num_classes, "Can't match # classes"

  # load video list
  with open(csv_file_name) as f:
    lines = [line.rstrip('\n') for line in f]

  # loop each video
  for line in lines:
    tokens = line.split(' ')
    # offset the id by -1
    video_path, label_id = tokens[0], int(tokens[1]) - 1
    video_path = video_path + '.mp4'
    video_item = {'filename': video_path,
                  'label': label_id,
                  'video_info': [],
                  'meta_label': []}
    video_list.append(video_item)

  return video_list, label_dict

def load_ant_hmdb51(csv_file_name, num_classes=51):
  """
  Parse CSV file for ucf101/hmdb51 dataset
  """
  # prep for outputs
  label_dict = {}
  video_list = []

  # read the csv file line by line
  with open(csv_file_name) as f:
    lines = [line.rstrip('\n') for line in f]

  # loop over each entry
  for line in lines:
    video_path, label_text, label_id = line.split(' ')
    video_path = video_path.replace('.avi', '.mp4')
    video_item = {'filename': video_path,
                  'label': int(label_id),
                  'video_info': [],
                  'meta_label': []}
    video_list.append(video_item)
    # json serialization won't take integer as keys
    label_dict[label_id] = label_text

  assert len(label_dict) == num_classes, "Can't match # classes"

  return video_list, label_dict

def load_ant_ucf101(csv_file_name, num_classes=101):
  """
  Parse CSV file for ucf101/hmdb51 dataset
  """
  # same format as hmdb51
  video_list, label_dict = load_ant_hmdb51(
    csv_file_name, num_classes=num_classes)
  return video_list, label_dict

def load_ant_20bn(json_file_name, label_file_name,
                  num_classes=174, is_test=False):
  """
  Parse json files for 20bn-v2 dataset
  """
  # load label dict
  with open(label_file_name) as f:
    label_dict = json.load(f)
  assert len(label_dict) == num_classes, "Can't match # classes"

  # load video file list
  with open(json_file_name) as f:
    video_item_list = json.load(f)

  # loop over each entry
  video_list = []
  object_name_list = []
  for video_item in video_item_list:
    # parse the item
    video_path = video_item['id'] + '.webm'
    if is_test:
      # only video id is available (skip the rest)
      new_video_item = {'filename': video_path,
                        'label': [],
                        'video_info': [],
                        'meta_label': []}
      video_list.append(new_video_item)
      continue

    label_text = video_item['template']
    # clean template
    label_text = label_text.replace('[', '')
    label_text = label_text.replace(']', '')
    label_id = int(label_dict[label_text])

    # create meta label info
    meta_label = {'caption' : video_item['label'],
                  'object_name' : video_item['placeholders']}
    new_video_item = {'filename': video_path,
                      'label': label_id,
                      'video_info': [],
                      'meta_label': meta_label}
    video_list.append(new_video_item)

    # additionally, add object name to list
    object_name_list.extend(video_item['placeholders'])

  # inver the label_dict (make it compatable with other datasets)
  label_dict = {v: k for k, v in label_dict.items()}

  return video_list, label_dict

# dummy functions (TO-DO)
def load_ant_moments():
  return NotImplemented

def load_ant_kinectics():
  return NotImplemented

def load_ant_actnet():
  return NotImplemented
