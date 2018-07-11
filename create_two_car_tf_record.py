r"""Convert raw Stanford Cars dataset to TFRecord for object_detection.

Example usage:
    python create_two_car_tf_record.py \
        --data_dir=/data/StanfordCars \
        --output_path=/home/HQ/dwetherby/workspace/stanford_cars/two_car_train.record \
	--label_map_path=two_car_label_map.pbtxt \
	--set=train
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import hashlib
import io
import logging

import PIL.Image

import scipy.io as sio

import numpy as np

import tensorflow as tf

from object_detection.utils import dataset_util
from object_detection.utils import label_map_util


flags = tf.app.flags

flags.DEFINE_string('data_dir','','Root directory to Stanford Cars dataset. (car_ims is a subfolder)')
flags.DEFINE_string('output_path','two_cars.tfrecord','Path to output TFRecord.')
flags.DEFINE_string('label_map_path','two_cars_label_map.pbtxt','Path to label map proto.')
flags.DEFINE_string('set','merged','Convert training set, test set, or merged set.')

FLAGS = flags.FLAGS

SETS = ['train', 'test', 'merged']

def dict_to_tf_example(annotation, dataset_directory, label_map_dict):
  im_path = str(np.squeeze(annotation['relative_im_path']))
  cls = np.squeeze(annotation['class'])
  if cls == '15':
    cls = '2'
  x1 = np.squeeze(annotation['bbox_x1'])
  y1 = np.squeeze(annotation['bbox_y1'])
  x2 = np.squeeze(annotation['bbox_x2'])
  y2 = np.squeeze(annotation['bbox_y2'])

  # read image
  full_img_path = os.path.join(dataset_directory, im_path)

  # read in the image and make a thumbnail of it
  max_size = 500, 500
  big_image = PIL.Image.open(full_img_path)
  big_image.thumbnail(max_size, PIL.Image.ANTIALIAS)
  width,height = big_image.size
  full_thumbnail_path = os.path.splitext(full_img_path)[0] + '_thumbnail.jpg'
  big_image.save(full_thumbnail_path)

  with tf.gfile.GFile(full_thumbnail_path, 'rb') as fid:
    encoded_jpg = fid.read()
  encoded_jpg_io = io.BytesIO(encoded_jpg)
  image = PIL.Image.open(encoded_jpg_io)

  xmin = []
  xmax = []
  ymin = []
  ymax = []

  # calculate box using original image coordinates
  xmin.append(x1 / width)
  xmax.append(x2 / width)
  ymin.append(y1 / height)
  ymax.append(y2 / height)

  # set width and height to thumbnail size for tfrecord ingest
  width,height = image.size

  classes = []
  classes_text = []

  label=''
  for name, val in label_map_dict.iteritems():
    if val == cls: 
      label = name
      break

  classes_text.append(label.encode('utf8'))
  classes.append(label_map_dict[label])
  
  example = tf.train.Example(features=tf.train.Features(feature={
	'image/height': dataset_util.int64_feature(height),
	'image/width': dataset_util.int64_feature(width),
	'image/filename': dataset_util.bytes_feature(full_img_path.encode('utf8')),
	'image/source_id': dataset_util.bytes_feature(full_img_path.encode('utf8')),
	'image/encoded': dataset_util.bytes_feature(encoded_jpg),
	'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
	'image/object/bbox/xmin': dataset_util.float_list_feature(xmin),
	'image/object/bbox/xmax': dataset_util.float_list_feature(xmax),
	'image/object/bbox/ymin': dataset_util.float_list_feature(ymin),
	'image/object/bbox/ymax': dataset_util.float_list_feature(ymax),
	'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
	'image/object/class/label': dataset_util.int64_list_feature(classes),
  }))
  return example 

def main(_):

  if FLAGS.set not in SETS:
    raise ValueError('set must be in : {}'.formats(SETS))

  train = FLAGS.set
  data_dir = FLAGS.data_dir

  writer = tf.python_io.TFRecordWriter(FLAGS.output_path)
  label_map_dict = label_map_util.get_label_map_dict(FLAGS.label_map_path)

  mat = sio.loadmat(os.path.join(data_dir,'cars_annos.mat'))

  for annotation in mat['annotations'][0]:
    test = np.squeeze(annotation['test'])
    if test:
      testset = 'test'
    else:
      testset = 'train'
 
    if train == 'merged' or train == testset:
      cls = np.squeeze(annotation['class'])
      if cls in (1,15):
        tf_example = dict_to_tf_example(annotation, data_dir, label_map_dict)
        writer.write(tf_example.SerializeToString())

  writer.close()

if __name__ == '__main__':
  tf.app.run()
