import os

import tensorflow as tf
import numpy as np

from . import generator
from . import critic
from . import helpers

class UGATIT:
  def __init__(self, **kwargs):
    self._base_ch = kwargs.get('base_ch', 64)
    self._gan_weight = kwargs.get('gan_weight', 1.)
    self._rec_weight = kwargs.get('rec_weight', 10.)
    self._ws_weight = kwargs.get('ws_weight', 10.)
    self._cam_weight = kwargs.get('cam_weight', 1000.)
    self._l2_regularization_weight = kwargs.get('l2_regularization_weight', 1e-4)
    self._ckpt = kwargs.get('ckpt', None)
    self._save_dir = kwargs.get('save_dir', 'ckpt')
    self._log_dir = kwargs.get('log_dir', 'log')
    self._gen_a_dir = kwargs.get('gen_a_dir', 'gen_a')
    self._gen_b_dir = kwargs.get('gen_b_dir', 'gen_b')
    self._log_step = kwargs.get('log_step', 100)
    self._init_lr = kwargs.get('init_lr', 1e-4)
    self._init_step = kwargs.get('init_step', 0)

    self._image_a_name = 'image_a'
    self._image_b_name = 'image_b'
    self._gen_a2b_name = 'gen_a2b'
    self._gen_b2a_name = 'gen_b2a'
    self._critic_a_name = 'critic_a'
    self._critic_b_name = 'critic_b'

    self._sess = None
    self._saver = None
    self._logger = None

  def fit_tf_data(self):
    feed = {self._lr: self._init_lr}
    step = self._init_step
    try:
      while True:
        image_a, image_b = self.sess.run([self._image_a, self._image_b])
        feed[self._image_a] = image_a
        feed[self._image_b] = image_b
        gen_b, gen_a, sum_str = self.sess.run([self._gen_a2b, self._gen_b2a, self._summary], feed)
        self.save('ckpt_%d' % step)
        self.log(sum_str)
        self.save_images(os.path.join(self._gen_a_dir, 'train_%s.jpg' % step), gen_a)
        self.save_images(os.path.join(self._gen_b_dir, 'train_%s.jpg' % step), gen_b)

        self.sess.run([self._train_c, self._train_g], feed)
        for i in range(self._log_step):
          self.sess.run([self._train_c, self._train_g], feed)
        step += self._log_step
        print("step: %d" % step, end='\r')
        
    except tf.errors.OutOfRangeError:
      print("Training Finished")

  def predict_tf_data(self):
    try:
      index = 0
      while True:
        image_a, image_b, gen_b, gen_a = self.sess.run(
            [self._image_a, self._image_b, self._gen_a2b, self._gen_b2a])
        self.save_images(os.path.join(self._gen_a_dir, 'ori_a_%s.jpg' % index), image_a)
        self.save_images(os.path.join(self._gen_b_dir, 'gen_b_%s.jpg' % index), gen_b)
        self.save_images(os.path.join(self._gen_b_dir, 'ori_b_%s.jpg' % index), image_b)
        self.save_images(os.path.join(self._gen_a_dir, 'gen_a_%s.jpg' % index), gen_a)
        index += 1
        print("index: %d" % index)

    except tf.errors.OutOfRangeError:
      print("Prediction Finished")

  def predict_a2b(self, batch_images):
    return self.sess.run(self._gen_a2b, {self._image_a: batch_images})

  def predict_b2a(self, batch_images):
    return self.sess.run(self._gen_b2a, {self._image_b: batch_images})

  def build(self, image_a, image_b, is_train=False):
    ''' big self time '''
    self._make_image_placeholder(image_a, image_b)
    self._normalize()
    self._make_generator_graph()
    self._make_critic_graph()
    if is_train:
      self._make_loss_graph()
      self._make_train_graph()
    self._summary = tf.summary.merge_all()

  def _make_image_placeholder(self, image_a, image_b):
    ''' self._image_a, self._image_b '''
    self._image_a = tf.placeholder_with_default(image_a, image_a.get_shape().as_list(), self._image_a_name)
    self._image_b = tf.placeholder_with_default(image_b, image_b.get_shape().as_list(), self._image_b_name)

  def _normalize(self):
    self._image_a = self._image_a * (1. / 127.5) - 1.
    self._image_b = self._image_b * (1. / 127.5) - 1.

  def _make_generator_graph(self):
    ''' self._gen_a2b, self._gen_b2a '''
    self._gen_a2b, self._cam_a2b, self._heat_a2b = generator.generator(
        self._image_a, self._base_ch, self._gen_a2b_name)
    self._gen_b2a, self._cam_b2a, self._heat_b2a = generator.generator(
        self._image_b, self._base_ch, self._gen_b2a_name)

    self._gen_a2a, self._cam_a2a, _ = generator.generator(self._image_a, self._base_ch, self._gen_b2a_name, True)
    self._gen_b2b, self._cam_b2b, _ = generator.generator(self._image_b, self._base_ch, self._gen_a2b_name, True)

    self._rec_a, _, _ = generator.generator(self._gen_a2b, self._base_ch, self._gen_b2a_name, True)
    self._rec_b, _, _ = generator.generator(self._gen_b2a, self._base_ch, self._gen_a2b_name, True)

  def _make_critic_graph(self):
    self._scores_a, self._cam_scores_a, self._heats_a = critic.critic(
        self._image_a, self._base_ch, self._critic_a_name)
    self._scores_b, self._cam_scores_b, self._heats_b = critic.critic(
        self._image_b, self._base_ch, self._critic_b_name)

    self._scores_b2a, self._cam_scores_b2a, self._heats_b2a = critic.critic(
        self._gen_b2a, self._base_ch, self._critic_a_name, True)
    self._scores_a2b, self._cam_scores_a2b, self._heats_a2b = critic.critic(
        self._gen_a2b, self._base_ch, self._critic_b_name, True)

  def _make_loss_graph(self):
    self._gen_a2b_loss_gan = critic.generator_loss(self._scores_a2b) + critic.generator_loss(self._cam_scores_a2b) 
    self._gen_a2b_loss_rec = helpers.l1_loss(self._image_b, self._rec_b)
    self._gen_a2b_loss_ws = helpers.l1_loss(self._image_b, self._gen_b2b)
    self._gen_a2b_loss_cam = generator.cam_loss(self._cam_a2b, self._cam_b2b)
    self._gen_a2b_loss = self._combine_gan_loss(
        self._gen_a2b_loss_gan, self._gen_a2b_loss_rec, self._gen_a2b_loss_ws, self._gen_a2b_loss_cam)

    self._gen_b2a_loss_gan = critic.generator_loss(self._scores_b2a) + critic.generator_loss(self._cam_scores_b2a)
    self._gen_b2a_loss_cam = generator.cam_loss(self._cam_b2a, self._cam_a2a)
    self._gen_b2a_loss_rec = helpers.l1_loss(self._image_a, self._rec_a)
    self._gen_b2a_loss_ws = helpers.l1_loss(self._image_a, self._gen_a2a)
    self._gen_b2a_loss = self._combine_gan_loss(
        self._gen_b2a_loss_gan, self._gen_b2a_loss_rec, self._gen_b2a_loss_ws, self._gen_b2a_loss_cam)

    self._generator_loss = self._gen_a2b_loss + self._gen_b2a_loss

    self._critic_a_loss = critic.critic_loss(self._scores_a, self._scores_b2a) + \
        critic.critic_loss(self._cam_scores_a, self._cam_scores_b2a)

    self._critic_b_loss = critic.critic_loss(self._scores_b, self._scores_a2b) + \
        critic.critic_loss(self._cam_scores_b, self._cam_scores_a2b)

    self._critic_loss = self._critic_a_loss + self._critic_b_loss

    self._gen_a2b_vars = [var for var in tf.trainable_variables() if var.name.startswith(self._gen_a2b_name)]
    self._gen_b2a_vars = [var for var in tf.trainable_variables() if var.name.startswith(self._gen_b2a_name)]
    self._generator_vars = self._gen_a2b_vars + self._gen_b2a_vars
    self._critic_a_vars = [var for var in tf.trainable_variables() if var.name.startswith(self._critic_a_name)]
    self._critic_b_vars = [var for var in tf.trainable_variables() if var.name.startswith(self._critic_b_name)]
    self._critic_vars = self._critic_a_vars + self._critic_b_vars
    self._all_vars = self._generator_vars + self._critic_vars
    self._all_weights = [var for var in self._all_vars if helpers.WEIGHT_NAME in var.name]
    self._l2_regularization = helpers.l2_regularization(self._all_weights, self._l2_regularization_weight)

    tf.summary.scalar('gen_a2b_loss', self._gen_a2b_loss)
    tf.summary.scalar('gen_b2a_loss', self._gen_b2a_loss)
    tf.summary.scalar('critic_a_loss', self._critic_a_loss)
    tf.summary.scalar('critic_b_loss', self._critic_b_loss)

  def _make_train_graph(self):
    ''' self._lr, self._train_g, self._train_c '''
    self._lr = tf.placeholder(tf.float32, shape=[], name='learning_rate')
    self._train_c = tf.train.AdamOptimizer(
        self._lr, beta1=0.5, beta2=0.999).minimize(
            self._critic_loss, var_list=self._critic_vars)
    self._train_g = tf.train.AdamOptimizer(
        self._lr, beta1=0.5, beta2=0.999).minimize(
            self._generator_loss, var_list=self._generator_vars)

  def _combine_gan_loss(self, loss_gan, loss_rec, loss_ws, loss_cam):
    loss = loss_gan * self._gan_weight + \
        loss_rec * self._rec_weight + \
        loss_ws * self._ws_weight + \
        loss_cam * self._cam_weight
    return loss

  @property
  def sess(self):
    if self._sess is None:
      self._init_sess()
    return self._sess

  def _init_sess(self):
    self._sess = tf.Session()
    if self._ckpt is not None:
      tf.train.Saver().restore(self._sess, self._ckpt)
    else:
      self._sess.run(tf.global_variables_initializer())

  def save(self, save_name='ckpt'):
    if self._saver is None:
      self._saver = tf.train.Saver(max_to_keep=25)
    save_path = os.path.join(self._save_dir, save_name)
    self._saver.save(self.sess, save_path)
    print("save %s" % save_path)

  def log(self, log_str, log_name='log'):
    if self._logger is None:
      self._logger = tf.summary.FileWriter(os.path.join(self._log_dir, log_name), self.sess.graph)
    self._logger.add_summary(log_str)

  def save_images(self, save_path, images):
    images = np.clip((images + 1.) * 127.5, 0, 255).astype(np.uint8)
    helpers.save_images_as_grid(save_path, images)
    print("save_images %s" % save_path)
