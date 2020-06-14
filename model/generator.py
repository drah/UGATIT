import tensorflow as tf

from . import modules
from . import helpers


def generator(net, base_ch, name, reuse=None, **kwargs):
  '''
  [N, H, W, 3]
      conv_7x7
  [N, H, W, base_ch]
      conv_3x3_s2
  [N, H//2, W//2, base_ch*2]
      conv_3x3_s2
  [N, H//4, W//4, base_ch*4]
      ins_norm_res_block_1, 2, 3, 4
  [N, H//4, W//4, base_ch*4]
      ...
      cam, gamma_beta_net
      ...
      conv_1x1
  [N, H//4, W//4, base_ch*4]
      adalin_res_block_1, 2, 3, 4
  [N, H//4, W//4, base_ch*4]
      resize and conv_3x3
  [N, H//2, W//2, base_ch*2]
      resize and conv_3x3
  [N, H, W, base_ch]
      conv_7x7
  [N, H, W, 3]
      tanh
  '''
  ch = base_ch
  with tf.variable_scope(name, reuse=reuse):
    net = modules.pad_conv_ins_relu(net, ch, (7, 7), (1, 1), 3, 3, 3, 3, 'pad_conv_ins_relu_1')

    ch *= 2
    net = modules.pad_conv_ins_relu(net, ch, (3, 3), (2, 2), 1, 1, 1, 1, 'pad_conv_ins_relu_2')

    ch *= 2
    net = modules.pad_conv_ins_relu(net, ch, (3, 3), (2, 2), 1, 1, 1, 1, 'pad_conv_ins_relu_3')

    for i in range(kwargs.get('n_res_block', 4)):
      net = modules.ins_norm_res_block(net, ch, 'ins_norm_res_block_%d' % i, reuse)

    net, cam_logit = modules.cam_block(net, 'cam_block', reuse) # (B, len//4, len//4, ch*2) ... ? net_gap == net_gmp
    
    net = helpers.conv2d(net, ch, (1, 1), (1, 1), 'VALID', 'conv2d_after_cam')
    net = tf.nn.relu(net)

    heatmap = helpers.compute_heatmap(net)

    shape = net.get_shape().as_list()
    flattened = helpers.flatten(net)
    flattened.set_shape([None, shape[1]*shape[2]*shape[3]])
    gamma, beta = modules.gamma_beta_net(flattened, ch, 'gamma_beta_net', reuse)
    gamma = helpers.expand_dims(gamma, [1, 2])
    beta = helpers.expand_dims(beta, [1, 2])

    for i in range(4):
      net = modules.adalin_res_block(net, ch, gamma, beta, 'adalin_res_block_%d' % i, reuse)

    for i in range(2) :
      ch //= 2
      net = helpers.resize_scale(net, 2)
      net = modules.pad_conv_ins_relu(net, ch, (3, 3), (1, 1), 1, 1, 1, 1, 'up_sampling_pad_conv_%d' % i, reuse)

    net = helpers.conv2d(net, 3, (7, 7), (1, 1), 'SAME', 'final_conv', reuse)
    net = tf.nn.tanh(net, 'generated_tanh_output')

    return net, cam_logit, heatmap
