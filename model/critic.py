import tensorflow as tf

from . import modules
from . import helpers

def critic(net, base_ch, name, reuse=None):
  with tf.variable_scope(name, reuse=reuse):
    local_logit, local_cam_logit, local_heatmap = _critic(net, base_ch, 3, 'local_critic', reuse)
    global_logit, global_cam_logit, global_heatmap = _critic(net, base_ch, 5, 'global_critic', reuse)

  logits = [local_logit, global_logit]
  cam_logits = [local_cam_logit, global_cam_logit]
  heatmaps = [local_heatmap, global_heatmap]
  return logits, cam_logits, heatmaps

def _critic(net, base_ch, n_base_layer, name, reuse=None):
  ch = base_ch
  with tf.variable_scope(name, reuse=reuse):
    for i in range(n_base_layer):
      net = modules.pad_spectral_conv(net, ch, (4, 4), (2, 2), 1, 1, 1, 1, 'spectral_conv_%d' % i, reuse=reuse)
      net = tf.nn.leaky_relu(net)
      ch *= 2

    net = modules.pad_spectral_conv(net, ch, (4, 4), (1, 1), 1, 2, 1, 2, 'final_spectral_conv', reuse=reuse)
    net = tf.nn.leaky_relu(net)
    ch *= 2

    net, cam_logit = modules.cam_block(net, 'cam_block', reuse, True)

    net = helpers.conv2d(net, ch, (1, 1), (1, 1), 'VALID', 'conv_1x1', reuse)
    net = tf.nn.leaky_relu(net)

    heatmap = helpers.compute_heatmap(net)
    
    net = modules.pad_spectral_conv(net, 1, (4, 4), (1, 1), 1, 2, 1, 2, 'critic_logit', reuse)
    return net, cam_logit, heatmap
