import os


class ModelDir:
  def __init__(self, root_dir):
    self.root_dir = root_dir
    self.gen_a_dir = os.path.join(root_dir, 'gen_a')
    self.gen_b_dir = os.path.join(root_dir, 'gen_b')
    self.ckpt_dir = os.path.join(root_dir, 'ckpt')
    self.log_dir = os.path.join(root_dir, 'log')
    for directory in [self.root_dir, self.gen_a_dir, self.gen_b_dir, self.ckpt_dir, self.log_dir]:
      os.makedirs(directory, exist_ok=True)