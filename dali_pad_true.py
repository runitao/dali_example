#!/usr/bin/env python
#encoding: utf8

import sys

from nvidia.dali import pipeline_def
import nvidia.dali.fn as fn
import nvidia.dali.types as types


@pipeline_def
def video_pipeline(filenames,
                   pad_sequences=True,
                   sequence_length=25,
                   stride=1,
                   shard_id=0,
                   num_shards=1,
                   device='gpu',
                   name='video_reader'):
  video = fn.readers.video(device=device,
                           filenames=filenames,
                           sequence_length=sequence_length,
                           shard_id=shard_id,
                           num_shards=num_shards,
                           random_shuffle=False,
                           initial_fill=1,
                           image_type=types.RGB,
                           dtype=types.FLOAT,
                           prefetch_queue_depth=2,
                           pad_sequences=pad_sequences,
                           file_list_include_preceding_frame=True,
                           stride=stride,
                           name=name)

  return video


def decode(filenames, pad=True):
  pipe = video_pipeline(filenames=filenames,
                        pad_sequences=pad,
                        batch_size=1,
                        sequence_length=25,
                        num_threads=1,
                        device_id=2,
                        device='gpu',
                        shard_id=0,
                        num_shards=1)
  pipe.build()
  for i in range(8):
    (video,) = pipe.run()


if __name__ == '__main__':
  filename = sys.argv[1] if len(sys.argv) > 1 else '437387963.mp4'

  print('decode with pad_sequences=False...')
  decode([filename], pad=False)
  print('decode with pad_sequences=False OK')

  print('\ndecode with pad_sequences=True...')
  decode([filename], pad=True)
  print('decode with pad_sequences=True OK')
