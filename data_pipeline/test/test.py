import unittest
import os

import tensorflow as tf
import numpy as np
import cv2

from ..factory import get_reader

class TestReader(unittest.TestCase):
  def setUp(self):
    self.data_dir = 'data_pipeline/test'
    self.result_dir = os.path.join(self.data_dir, 'result')
    self.image_paths = [
        os.path.join(self.data_dir, 'img_1.jpg'),
        os.path.join(self.data_dir, 'img_2.jpg')]
    self.batch_size = 10
    self.imm_hw = (143, 143)
    self.dest_hw = (128, 128)
    self.repeat_count = 5

  def test_tf_read_with_aug(self):
    dataset = get_reader('tf_read_with_aug')(
        self.image_paths,
        batch_size=self.batch_size,
        imm_hw=self.imm_hw,
        dest_hw=self.dest_hw,
        repeat_count=self.repeat_count)
    with tf.Session() as sess:
      batch_images = sess.run(dataset)
    self.assertEqual(batch_images.shape[0], self.batch_size)
    self.assertEqual(batch_images.shape[1:3], self.dest_hw)
    self.assertEqual(batch_images.dtype, np.float32)
    self.assertAlmostEqual(batch_images.max(), 255, delta=10)
    for i, image in enumerate(batch_images):
      cv2.imwrite(os.path.join(self.result_dir, 'tf_read_with_aug_%d.jpg' % i), image[:,:,::-1])

  def test_tf_read(self):
    dataset = get_reader('tf_read')(
        self.image_paths,
        batch_size=self.batch_size,
        dest_hw=self.dest_hw,
        repeat_count=self.repeat_count)
    with tf.Session() as sess:
      batch_images = sess.run(dataset)
    self.assertEqual(batch_images.shape[0], self.batch_size)
    self.assertEqual(batch_images.shape[1:3], self.dest_hw)
    self.assertEqual(batch_images.dtype, np.float32)
    self.assertAlmostEqual(batch_images.max(), 255, delta=10)
    for i, image in enumerate(batch_images):
      cv2.imwrite(os.path.join(self.result_dir, 'rf_read_%d.jpg' % i), image[:,:,::-1])



if __name__ == '__main__':
  unittest.main()