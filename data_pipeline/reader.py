import glob
import os

import tensorflow as tf
from . import tf_helper


def get_paths(data_dir):
  return glob.glob(os.path.join(data_dir, '*'))


def tf_read_with_aug(
    image_paths,
    batch_size=50,
    buffer_size=100,
    imm_hw=(286, 286),
    dest_hw=(256, 256),
    repeat_count=None,
    drop_remainder=False,
    **kwargs):

  # image_paths could be path to dir or list of image paths
  if os.path.isdir(image_paths):
    image_paths = get_paths(image_paths)

  dataset = tf.data.Dataset.from_tensor_slices(image_paths)
  dataset = dataset.repeat(repeat_count)
  dataset = dataset.shuffle(buffer_size)
  dataset = dataset.map(tf_helper.read_image)

  dataset = dataset.map(tf.image.random_flip_left_right)
  dataset = dataset.map(lambda x: tf_helper.resize_images(x, imm_hw))
  dataset = dataset.map(lambda x: tf_helper.random_crop(x, dest_hw))

  dataset = dataset.batch(batch_size, drop_remainder)
  dataset = dataset.make_one_shot_iterator().get_next()
  return dataset


def tf_read(
    image_paths,
    batch_size=50,
    buffer_size=100,
    dest_hw=(256, 256),
    repeat_count=None,
    drop_remainder=False,
    **kwargs):

  if os.path.isdir(image_paths):
    image_paths = get_paths(image_paths)

  dataset = tf.data.Dataset.from_tensor_slices(image_paths)
  dataset = dataset.repeat(repeat_count)
  dataset = dataset.shuffle(buffer_size)
  dataset = dataset.map(tf_helper.read_image)

  dataset = dataset.map(tf.image.random_flip_left_right)
  dataset = dataset.map(lambda x: tf_helper.resize_images(x, dest_hw))

  dataset = dataset.batch(batch_size, drop_remainder)
  dataset = dataset.make_one_shot_iterator().get_next()
  return dataset


def tf_read_with_more_aug(
    image_paths,
    batch_size=50,
    buffer_size=100,
    imm_hw=(286, 286),
    dest_hw=(256, 256),
    repeat_count=None,
    drop_remainder=False,
    **kwargs):

  raise NotImplementedError

  if os.path.isdir(image_paths):
    image_paths = get_paths(image_paths)

  dataset = tf.data.Dataset.from_tensor_slices(image_paths)
  dataset = dataset.repeat(repeat_count)
  dataset = dataset.shuffle(buffer_size)
  dataset = dataset.map(tf_helper.read_image)

  dataset = dataset.map(tf.image.random_flip_left_right)
  dataset = dataset.map(lambda x: tf_helper.resize_images(x, imm_hw))
  # random hue
  # random brightness
  # random contrast
  # random saturation
  # random jpeg_quality
  # random rotate
  # central crop
  dataset = dataset.map(lambda x: tf_helper.random_crop(x, dest_hw))

  dataset = dataset.batch(batch_size, drop_remainder)
  dataset = dataset.make_one_shot_iterator().get_next()
  return dataset