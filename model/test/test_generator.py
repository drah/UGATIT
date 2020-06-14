import unittest

import tensorflow as tf
from ..generator import generator


class TestGenerator(unittest.TestCase):
  def setUp(self):
    self.net = tf.placeholder(tf.float32, [None, 256, 256, 3])
    self.base_ch = 64
    self.out_shape = self.net.get_shape().as_list()
    return super().setUp()

  def test_generator(self):
    out = generator(self.net, self.base_ch, 'generator')
    self.assertEqual(out.get_shape().as_list(), self.out_shape)


if __name__ == '__main__':
  unittest.main()