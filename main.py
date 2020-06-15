import os
import argparse
from datetime import datetime

import data_pipeline
import model
import helper


def main(args):
  model_dir = helper.directory.ModelDir(args.save_dir)

  reader_args = {
      'batch_size': args.batch_size,
      'buffer_size': args.buffer_size,
      'imm_hw': (args.imm_h, args.imm_w),
      'dest_hw': (args.dest_h, args.dest_w),
      'repeat_count': args.repeat_count,
      'drop_remainder': bool(args.drop_remainder)}

  model_args = {
      'base_ch': args.base_ch,
      'gan_weight': args.gan_weight,
      'rec_weight': args.rec_weight,
      'ws_weight': args.ws_weight,
      'cam_weight': args.cam_weight,
      'l2_regularization_weight': args.l2_regularization_weight,
      'ckpt': args.ckpt,
      'save_dir': model_dir.ckpt_dir,
      'log_dir': model_dir.log_dir,
      'gen_a_dir': model_dir.gen_a_dir,
      'gen_b_dir': model_dir.gen_b_dir,
      'log_step': args.log_step,
      'init_lr': args.init_lr,
      'init_step': args.init_step}

  m = model.factory.get_model(args.model)(**model_args)

  if args.phase == 'train':
    tr_read_fn = data_pipeline.get_reader(args.train_reader)
    train_a = tr_read_fn(args.train_A_data_dir, **reader_args)
    train_b = tr_read_fn(args.train_B_data_dir, **reader_args)
    m.build(train_a, train_b, True)
    m.fit_tf_data()

  elif args.phase == 'test':
    te_read_fn = data_pipeline.get_reader(args.test_reader)
    test_a = tr_read_fn(args.test_A_data_dir, **reader_args)
    test_b = tr_read_fn(args.test_B_data_dir, **reader_args)
    m.build(test_a, test_b)
    m.predict_tf_data()

  elif args.phase == 'predict':
    raise NotImplementedError


if __name__ == '__main__':
  parser = argparse.ArgumentParser('UGATIT')

  parser.add_argument('--train_A_data_dir', default='./data/selfie2anime/trainA/',
      help='path to train A data dir')
  parser.add_argument('--train_B_data_dir', default='./data/selfie2anime/trainB/',
      help='path to train B data dir')
  parser.add_argument('--test_A_data_dir', default='./data/selfie2anime/testA/',
      help='path to test A data dir')
  parser.add_argument('--test_B_data_dir', default='./data/selfie2anime/testB/',
      help='path to test B data dir')

  parser.add_argument('--train_reader', default='tf_read_with_aug',
      help='tf_read, tf_read_with_aug')
  parser.add_argument('--test_reader', default='tf_read',
      help='tf_read, tf_read_with_aug')

  parser.add_argument('--batch_size', dest='batch_size', type=int, default=100,
      help='the size of each batch for training')
  parser.add_argument('--buffer_size', type=int, default=100)
  parser.add_argument('--imm_h', type=int, default=286)
  parser.add_argument('--imm_w', type=int, default=286)
  parser.add_argument('--dest_h', type=int, default=256)
  parser.add_argument('--dest_w', type=int, default=256)
  parser.add_argument('--repeat_count', type=int, default=1)
  parser.add_argument('--drop_remainder', type=int, default=0, help='1: True, 0: False')

  parser.add_argument('--model', default='UGATIT', help='th/bate name of the model')
  parser.add_argument('--ckpt', default=None, help='checkpoint of model')
  parser.add_argument('--save_dir', default='ugatit',
      help='the directory for saving training logs and checkpoints')

  parser.add_argument('-init_step', dest='start_step', type=int, default=0,
      help='the number from which training step starts')
  parser.add_argument('--base_ch', dest='base_ch', type=int, default=64)
  parser.add_argument('--gan_weight', dest='gan_weight', type=float, default=1)
  parser.add_argument('--rec_weight', dest='rec_weight', type=float, default=10)
  parser.add_argument('--ws_weight', dest='ws_weight', type=float, default=10)
  parser.add_argument('--cam_weight', dest='cam_weight', type=float, default=1000)
  parser.add_argument('--l2_regularization_weight', dest='l2_regularization_weight', type=float, default=1e-4)
  parser.add_argument('--log_step', dest='log_step', type=int, default=100)
  parser.add_argument('--init_lr', dest='init_lr', type=float, default=1e-4)
  parser.add_argument('--init_step', dest='init_step', type=int, default=0)

  parser.add_argument('--phase', dest='phase', type=str, default='train',
      help='train, test, predict')

  args = parser.parse_args()
  main(args)