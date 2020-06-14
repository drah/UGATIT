import unittest

import tensorflow as tf
from .. import model


class TestModel(unittest.TestCase):
  def setUp(self):
    self.image_a = tf.placeholder(tf.float32, [None, 256, 256, 3])
    self.image_b = tf.placeholder(tf.float32, [None, 256, 256, 3])
    self.base_ch = 64
    return super().setUp()

  def test_model(self):
    m = model.UGATIT()
    m.build(self.image_a, self.image_b)


if __name__ == '__main__':
  unittest.main()