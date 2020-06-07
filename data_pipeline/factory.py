from . import reader

def get_reader(reader_name):
  return {
      'tf_read_with_aug': reader.tf_read_with_aug,
      'tf_read': reader.tf_read,
      'tf_read_with_more_aug': reader.tf_read_with_more_aug,
  }[reader_name]

  
