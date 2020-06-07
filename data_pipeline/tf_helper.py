import tensorflow as tf 


def read_image(image_path):
  content = tf.read_file(image_path)
  image = tf.image.decode_image(content)
  return image

def resize_images(images, dest_hw):
  images = tf.cast(images, tf.float32)
  if images.shape.rank == 3:
    images = tf.expand_dims(images, 0)
    images = _resize_images_if_need(images, dest_hw)
    images = images[0]
  else:
    images = _resize_images_if_need(images, dest_hw)

  return images

def _resize_images_if_need(images, dest_hw):
  resized = tf.image.resize(images, dest_hw)
  images = tf.cond(
      tf.reduce_all(tf.equal(tf.shape(images)[1:3], dest_hw)),
      lambda: images,
      lambda: resized)
  return images

def random_crop(images, dest_hw):
  if len(images.shape) == 3:
    images = tf.image.random_crop(images, tf.concat([dest_hw, tf.shape(images)[-1:]], 0))
  else:
    shape = tf.shape(images)
    images = tf.image.random_crop(images, tf.concat([shape[0:], dest_hw, shape[-1:], 0]))

  return images