import tensorflow as tf

from . import helpers

def pad_conv_ins(
    node, n_filter, k_size, strides, pad_up, pad_down, pad_left, pad_right, name, reuse=None):
  with tf.variable_scope(name, reuse=reuse):
    node = helpers.reflect_pad(node, pad_up, pad_down, pad_left, pad_right)
    node = helpers.conv2d(node, n_filter, k_size, strides, 'VALID', 'conv_1')
    node = helpers.instance_norm(node, name='instance_norm_1')
  return node

def pad_conv_ins_relu(
    node, n_filter, k_size, strides, pad_up, pad_down, pad_left, pad_right, name, reuse=None):
  node = pad_conv_ins(node, n_filter, k_size, strides, pad_up, pad_down, pad_left, pad_right, name, reuse)
  node = tf.nn.relu(node)
  return node

def pad_conv_ins_lrelu(
    node, n_filter, k_size, strides, pad_up, pad_down, pad_left, pad_right, name, reuse=None):
  node = pad_conv_ins(node, n_filter, k_size, strides, pad_up, pad_down, pad_left, pad_right, name, reuse)
  node = tf.nn.leaky_relu(node)
  return node

def ins_norm_res_block(node, n_filter, name, reuse=None):
  ori = node
  with tf.variable_scope(name, reuse=reuse):
    node = pad_conv_ins_relu(node, n_filter, (3, 3), (1, 1), 1, 1, 1, 1, 'res_1', reuse)
    node = pad_conv_ins(node, n_filter, (3, 3), (1, 1), 1, 1, 1, 1, 'res_2', reuse)
  return ori + node

def pad_conv_ins(
    node, n_filter, k_size, strides, pad_up, pad_down, pad_left, pad_right, name, reuse=None):
  with tf.variable_scope(name, reuse=reuse):
    node = helpers.reflect_pad(node, pad_up, pad_down, pad_left, pad_right)
    node = helpers.conv2d(node, n_filter, k_size, strides, 'VALID', 'conv_1')
    node = helpers.instance_norm(node, name='instance_norm_1')
  return node

def cam(node, name, reuse=None, apply_spectral_norm=False):
  node, w, b = helpers.dense(node, 1, name, reuse, True)
  return node, w + b

def cam_block(net, name, reuse=None, apply_spectral_norm=False):
  with tf.variable_scope(name, reuse=reuse):
    gap = helpers.global_average_pooling(net) # (B, 1, 1, ch)
    cam_logit_gap, cam_weight_gap = cam(gap, 'cam_logit', reuse, apply_spectral_norm) # (B, 1), (ch, 1)
    net_gap = net * tf.squeeze(cam_weight_gap, 1) # (B, h', w', ch) * (ch) = (B, h', w', ch)

    gmp = helpers.global_max_pooling(net) # (B, 1, 1, ch)
    cam_logit_gmp, cam_weight_gmp = cam(gmp, 'cam_logit', True, apply_spectral_norm) # (B, 1), (ch, 1)
    net_gmp = net * tf.squeeze(cam_weight_gmp, 1) # (B, h', w', ch) * (ch) = (B, h', w', ch) ... ? cam_weight_gmp == cam_weight_gap

    cam_logit = tf.concat([cam_logit_gap, cam_logit_gmp], -1) # (B, 2)
    net = tf.concat([net_gap, net_gmp], -1) # (B, len//4, len//4, ch*2) ... ? net_gap == net_gmp
  return net, cam_logit

def gamma_beta_net(node, n, name, reuse=None):
  with tf.variable_scope(name, reuse=reuse):
    node = helpers.dense(node, n, 'dense_1', reuse)
    node = tf.nn.relu(node)

    node = helpers.dense(node, n, 'dense_2', reuse)
    node = tf.nn.relu(node)

    gamma = helpers.dense(node, n, 'dense_gamma', reuse)

    beta = helpers.dense(node, n, 'dense_beta', reuse)

  return gamma, beta

def adaptive_instance_layer_norm(node, gamma, beta, eps=1e-5, name=None, reuse=None):
  with tf.variable_scope(name, reuse=reuse):
    norm_per_ch, _, _ = helpers.standardize(node, [1, 2], name='standardize_per_channel')
    norm_per_layer, _, _ = helpers.standardize(node, [1, 2, 3], name='standardize_per_layer')
    rho = helpers.get_01_trainable([node.shape[-1]], 'rho', init_value=1.)

    # ?
    rho = tf.clip_by_value(rho - tf.constant(0.1), 0.0, 1.0)

    interpolate = tf.add(rho * norm_per_ch, (1. - rho) * norm_per_layer, name='interpolate')
    adalin = tf.add(interpolate * gamma, beta, name='adalin')
  return adalin

def pad_conv_adalin_relu(
    node, n_filter, k_size, strides, pad_up, pad_down, pad_left, pad_right, gamma, beta, name, reuse=None):
  node = pad_conv_adalin(
      node, n_filter, k_size, strides, pad_up, pad_down, pad_left, pad_right, gamma, beta, name, reuse)
  node = tf.nn.relu(node)
  return node

def pad_conv_adalin(
    node, n_filter, k_size, strides, pad_up, pad_down, pad_left, pad_right, gamma, beta, name, reuse=None):
  with tf.variable_scope(name, reuse=reuse):
    node = helpers.reflect_pad(node, pad_up, pad_down, pad_left, pad_right)
    node = helpers.conv2d(node, n_filter, k_size, strides, 'VALID', 'conv_1', reuse=reuse)
    node = adaptive_instance_layer_norm(node, gamma, beta, name='adalin', reuse=reuse)
  return node

def adalin_res_block(node, n_filter, gamma, beta, name, reuse=None):
  ori = node
  with tf.variable_scope(name, reuse=reuse):
    node = pad_conv_adalin_relu(node, n_filter, (3, 3), (1, 1), 1, 1, 1, 1, gamma, beta, 'res_1', reuse)
    node = pad_conv_adalin(node, n_filter, (3, 3), (1, 1), 1, 1, 1, 1, gamma, beta, 'res_2', reuse)
  return ori + node

def pad_spectral_conv(
    node, n_filter, k_size, strides, pad_up, pad_down, pad_left, pad_right, name, reuse=None):
  with tf.variable_scope(name, reuse=reuse):
    node = helpers.reflect_pad(node, pad_up, pad_down, pad_left, pad_right)
    node = helpers.spectral_conv2d(node, n_filter, k_size, strides, 'VALID', 'conv_1')
  return node
    
