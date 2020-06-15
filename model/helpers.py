import os
from math import sqrt

import tensorflow as tf 
import cv2

WEIGHT_NAME = 'weight'
BIAS_NAME = 'bias'

def _dense(node, n, name=None, reuse=None, apply_spectral_norm=False):
  with tf.variable_scope(name, reuse=reuse):
    w = get_weight([node.shape[-1], n], 'dense_' + WEIGHT_NAME)
    if apply_spectral_norm:
      w = spectral_norm(w, 'spectral_norm', reuse)
    b = get_bias([n], name='dense_' + BIAS_NAME)

  with tf.name_scope(name):
    node = tf.matmul(node, w) + b

  return node, w, b

def dense(node, n, name=None, reuse=None, also_return_weight_bias=False):
  node, w, b = _dense(node, n, name, reuse)
  if also_return_weight_bias:
    return node, w, b
  else:
    return node

def conv2d(node, n_filter, k_size, strides, padding, name=None, reuse=None):
  with tf.variable_scope(name, reuse=reuse):
    w = get_weight([k_size[0], k_size[1], node.shape[-1], n_filter], name='conv2d_' + WEIGHT_NAME)
    b = get_bias([n_filter], name='conv2d_' + BIAS_NAME)

    node = tf.nn.conv2d(node, w, [1, strides[0], strides[1], 1], padding)
    node = tf.nn.bias_add(node, b)

  return node

def optimize(loss, learning_rate=2e-4, decay_steps=10000, decay_rate=0.9, var_list=None, name=None):
  with tf.variable_scope(name or 'optimizer'):
    global_step = get_global_step()
    dlr = tf.train.exponential_decay(learning_rate, global_step, decay_steps, decay_rate, True)
    opt = tf.train.AdamOptimizer(dlr, beta1=0.5, beta2=0.999)
    train_step = opt.minimize(loss, global_step, var_list)
  return global_step, train_step

def get_global_step():
  return tf.get_variable('global_step', shape=[], initializer=tf.initializers.constant(0, tf.int64), trainable=False)

def get_weight(shape, name=None):
  w = tf.get_variable(name or WEIGHT_NAME, shape, initializer=tf.initializers.he_uniform())
  return w

def get_bias(shape, init_value=0.2, name=None):
  b = tf.get_variable(name or BIAS_NAME, shape, initializer=tf.initializers.constant(init_value))
  return b

def get_saver(var_list=None, max_to_keep=10):
  return tf.train.Saver(var_list, max_to_keep=max_to_keep)

def handle_dir(path):
  if not os.path.exists(path):
    os.mkdir(path)

def l2_loss(label, pred):
  return tf.reduce_mean(tf.squared_difference(label, pred))

def metric(label, pred):
  return tf.sqrt(l2_loss(label, pred))

def batch_norm(node, is_train, decay_rate=0.96, name='batch_norm'):
  # not tested
  depth = node.shape[-1]
  with tf.variable_scope(name):
    moving_mean = tf.get_variable('moving_mean', [depth], initializer=tf.initializers.zeros(), trainable=False)
    moving_var = tf.get_variable('moving_var', [depth], initializer=tf.initializers.ones(), trainable=False)
    batch_mean, batch_var = tf.nn.moments(node, [i for i in range(len(node.shape)-1)])
    scale = tf.constant(1.)
    offset = tf.constant(0.)
    eps = tf.constant(1e-3)

    inc_rate = 1. - decay_rate
    update_moving_mean = tf.assign(moving_mean, moving_mean * decay_rate + batch_mean * inc_rate)
    update_moving_var = tf.assign(moving_var, moving_var * decay_rate + batch_var * inc_rate)

    def is_train_true():
      with tf.control_dependencies([update_moving_mean, update_moving_var]):
        return tf.nn.batch_normalization(node, batch_mean, batch_var, offset, scale, eps)

    def is_train_false():
      return tf.nn.batch_normalization(node, moving_mean, moving_var, offset, scale, eps)
    
    node = tf.cond(is_train, is_train_true, is_train_false)
  return node

def reflect_pad(node, pad_up, pad_down, pad_left, pad_right):
  return tf.pad(node, [[0, 0], [pad_up, pad_down], [pad_left, pad_right], [0, 0]], mode='REFLECT')

def instance_norm(node, eps=1e-5, center=True, scale=True, name=None, reuse=None):
  return tf.contrib.layers.instance_norm(
      node, epsilon=eps, center=center, scale=scale, scope=name)

def global_average_pooling(node):
  return tf.reduce_mean(node, [1, 2])

def global_max_pooling(node):
  return tf.reduce_max(node, [1, 2])

def flatten(node, name='flatten'):
  with tf.name_scope(name):
    shape = tf.shape(node)
    flattened = tf.reshape(node, tf.stack([shape[0], -1]))
  return flattened

def get_01_trainable(shape, name, init_value=1.0):
  return tf.get_variable(
      name,
      shape,
      initializer=tf.constant_initializer(init_value),
      constraint=lambda x: tf.clip_by_value(x, 0., 1.))

def standardize(node, axis, eps=1e-5, name='standardize'):
  with tf.name_scope(name):
    mean, var = tf.nn.moments(node, axis, keep_dims=True)
    standardized = tf.multiply(node - mean, tf.rsqrt(var + eps), name='standardize')
  return standardized, mean, var

def expand_dims(node, dims):
  for d in dims:
    node = tf.expand_dims(node, d)
  return node

def resize_scale(node, scale):
  shape = node.get_shape().as_list()
  new_size = int(shape[1] * scale), int(shape[2] * scale)
  return tf.image.resize_nearest_neighbor(node, new_size)

def compute_heatmap(node, name='compute_heatmap'):
  with tf.name_scope(name):
    heatmap = tf.reduce_sum(node, -1)
  return heatmap

def spectral_conv2d(node, n_filter, k_size, strides, padding, name=None, reuse=None):
  with tf.variable_scope(name, reuse=reuse):
    w = get_weight([k_size[0], k_size[1], node.shape[-1], n_filter], name='conv2d_' + WEIGHT_NAME)
    w = spectral_norm(w, 'conv2d_%s_spectral_norm' % WEIGHT_NAME, reuse=reuse)
    b = get_bias([n_filter], name='conv2d_' + BIAS_NAME)

    node = tf.nn.conv2d(node, w, [1, strides[0], strides[1], 1], padding)
    node = tf.nn.bias_add(node, b)

  return node

def spectral_dense(node, n, name=None, reuse=None, also_return_weight_bias=False):
  node, w, b = _dense(node, n, name, reuse, True)
  if also_return_weight_bias:
    return node, w, b
  else:
    return node

def power_iteration(u, weight):
  v = tf.matmul(u, tf.transpose(weight)) # (1, N) * (N, M) = (1, M)
  v_norm = tf.nn.l2_normalize(v)
  u = tf.matmul(v_norm, weight) # (1, M) * (M, N) = (1, N)
  u_norm = tf.nn.l2_normalize(u)

  u_norm = tf.stop_gradient(u_norm)
  v_norm = tf.stop_gradient(v_norm)
  return v_norm, u_norm

def spectral_norm(weight, name, reuse=None):
  shape = weight.get_shape().as_list()
  with tf.variable_scope(name, reuse=reuse):
    weight = tf.reshape(weight, [-1, shape[-1]]) # (M, N)
    u = tf.get_variable("u", [1, shape[-1]], initializer=tf.random_normal_initializer(), trainable=False)

    v_norm, u_norm = power_iteration(u, weight)

    left = tf.matmul(v_norm, weight) # (1, M) * (M, N) = (1, N)
    sigma = tf.matmul(left, tf.transpose(u_norm)) # (1, N) * (N, 1) = (1, 1)

    with tf.control_dependencies([tf.assign(u, u_norm)]):
      weight_norm = weight * tf.reciprocal(sigma)
      weight_norm = tf.reshape(weight_norm, shape)

  return weight_norm

def l1_loss(val_1, val_2):
  return tf.reduce_mean(tf.abs(val_1 - val_2))

def l2_regularization(vars, reg_weight):
  return tf.add_n([tf.nn.l2_loss(var) for var in vars]) * reg_weight

def save_images_as_grid(save_path, images):
  ''' images: np.array with dtype uint8 and shape (N, H, W, C)'''
  shape = images.shape
  length = int(sqrt(shape[0]))
  grid_images = images[:length*length,:,:,::-1]
  grid_images = grid_images.reshape([length, length, shape[1], shape[2], shape[3]])
  grid_images = grid_images.transpose([0, 2, 1, 3, 4])
  grid_images = grid_images.reshape([length * shape[1], length * shape[2], shape[3]])
  cv2.imwrite(save_path, grid_images)
