import unittest

import tensorflow as tf
from ..critic import critic

class TestCritic(unittest.TestCase):
  def setUp(self):
    self.net = tf.placeholder(tf.float32, [None, 256, 256, 3])
    self.base_ch = 64
    self.out_shape = [[None, 32, 32, 1], [None, 8, 8, 1]]
    self.cam_shape = [[None, 2], [None, 2]]
    self.heatmap_shape = [[None, 32, 32], [None, 8, 8]]
    return super().setUp()

  def test_critic(self):
    out, cam, heatmap = critic(self.net, self.base_ch, 'critic')
    for i in range(2):
      self.assertEqual(out[i].get_shape().as_list(), self.out_shape[i])
      self.assertEqual(cam[i].get_shape().as_list(), self.cam_shape[i])
      self.assertEqual(heatmap[i].get_shape().as_list(), self.heatmap_shape[i])


if __name__ == '__main__':
  unittest.main()
