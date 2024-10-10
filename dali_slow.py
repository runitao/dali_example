#!/usr/bin/env python
#encoding: utf8

import sys
import numpy as np
import time

from nvidia.dali import pipeline_def
import nvidia.dali.fn as fn
import nvidia.dali.types as types

@pipeline_def
def video_pipeline(filenames,
                   labels=[],
                    sequence_length=16,
                    stride=1,
                    shard_id=0,
                    num_shards=1,
                    device='gpu',
                    name='video_reader'):
  video, label, start_frame_num, timestamps = fn.readers.video(
      device=device,
      filenames=filenames,
      labels=labels,
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
      file_list_include_preceding_frame=True,
      pad_last_batch=True,
      pad_sequences=True,
      stride=stride,
      name=name)

  return timestamps



def decode(filename, iteration):
  batch_size =1
  sequence_length=25 # video avg_frame_rate=25/1
  pipe = video_pipeline(filenames=[filename],batch_size=batch_size,
                        sequence_length=sequence_length, num_threads=1,
                        device_id=0, device='gpu',
                        shard_id=0, num_shards=1)
  pipe.build()

  if iteration == -1:
    epoch_size = pipe.epoch_size('video_reader')
    iteration = (epoch_size + batch_size - 1) // batch_size  # 计算需要的迭代次数
  num_frames = 0
  start = time.monotonic()
  for i in range(iteration):
    (timestamps,) = pipe.run()
    ts = timestamps.as_cpu().as_array()[0]
    num_frames += len(ts)
    # print(ts)
    if ts[-1] == -1:
      num_frames -= np.count_nonzero(ts == -1)

  end = time.monotonic()
  elapsed = end - start

  print(f"\telapsed={elapsed:.2f}, fps={num_frames/elapsed:.2f}, num_frames={num_frames}")


if __name__ == '__main__':
  filename = sys.argv[1] if len(sys.argv) > 1 else '439579348.mp4'

  print("deocde the first 5s of the video")
  decode(filename, 5)

  print("deocde the first 10s of the video")
  decode(filename, 10)

  print("deocde the first 20s of the video")
  decode(filename, 20)

  print("deocde the first 50s of the video")
  decode(filename, 50)

  print("deocde all the video")
  decode(filename, -1)
