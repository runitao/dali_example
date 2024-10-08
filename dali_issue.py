#!/usr/bin/env python
#encoding: utf8

import os
import sys
import numpy as np
import tempfile
import random
import shutil
from PIL import Image

from nvidia.dali import pipeline_def
import nvidia.dali.fn as fn
import nvidia.dali.types as types


@pipeline_def
def video_pipeline(file_list,
                    sequence_length=16,
                    resize_shorter=256.0,
                    crop=[224, 224],
                    stride=1,
                    shard_id=0,
                    num_shards=1,
                    device='gpu',
                    name='video_reader'):
  video, label, start_frame_num, timestamps = fn.readers.video(
      device=device,
      file_list=file_list,
      sequence_length=sequence_length,
      shard_id=shard_id,
      num_shards=num_shards,
      random_shuffle=False,
      initial_fill=1,
      image_type=types.RGB,
      dtype=types.FLOAT,
      file_list_frame_num=True,
      enable_frame_num=True,
      enable_timestamps=True,
      prefetch_queue_depth=2,
      stride=stride,
      name=name)

  return video, label


batch_size=4

file_list = sys.argv[1] if len(sys.argv) > 1 else 'filelist.txt'
pipe = video_pipeline(file_list=file_list,batch_size=batch_size,
                      sequence_length=16, num_threads=1,
                      device_id=0, device='gpu',
                      shard_id=0, num_shards=1)
pipe.build()
for i in range(100000):
  video, label = pipe.run()
  print(f"iter {i} label {label.as_cpu().as_array().squeeze()}")
