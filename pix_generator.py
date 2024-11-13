import os
import tensorflow as tf

# downsampling module using convolutions
def downsample(filters, size, apply_batchnorm=True):
  
  #kernel initializer
  initializer = tf.random_normal_initializer(0., 0.02)


  result = tf.keras.Sequential()
  result.add(
      tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                             kernel_initializer=initializer, use_bias=False))

  if apply_batchnorm:
    result.add(tf.keras.layers.BatchNormalization())

  result.add(tf.keras.layers.LeakyReLU())

  return result


def upsample(filters, size, apply_dropout=False):
  initializer = tf.random_normal_initializer(0., 0.02)

  result = tf.keras.Sequential()
  result.add(
    tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                    padding='same',
                                    kernel_initializer=initializer,
                                    use_bias=False))

  result.add(tf.keras.layers.BatchNormalization())

  if apply_dropout:
      result.add(tf.keras.layers.Dropout(0.5))

  result.add(tf.keras.layers.ReLU())

  return result

# definition of generator
def Generator(in_channels, patch_dim, out_channels):
  inputs = tf.keras.layers.Input(shape=[patch_dim[0], patch_dim[1], in_channels])

  down_stack = [
    downsample(64, 4, apply_batchnorm=False),  # (bs, 128, 128, 64)
    downsample(128, 4),  # (bs, 64, 64, 128)
    downsample(256, 4),  # (bs, 32, 32, 256)
    downsample(512, 4),  # (bs, 16, 16, 512)
    downsample(512, 4),  # (bs, 8, 8, 512)
    downsample(512, 4),  # (bs, 4, 4, 512)
    # downsample(512, 4),  # (bs, 2, 2, 512)
    # downsample(512, 4),  # (bs, 1, 1, 512)
  ]

  up_stack = [
    upsample(512, 4, apply_dropout=True),  # (bs, 2, 2, 1024)
    upsample(512, 4, apply_dropout=True),  # (bs, 4, 4, 1024)
    upsample(256, 4),  # (bs, 8, 8, 1024)
    upsample(128, 4),  # (bs, 16, 16, 1024)
    upsample(64, 4),  # (bs, 32, 32, 512)
    # upsample(128, 4),  # (bs, 64, 64, 256)
    # upsample(64, 4),  # (bs, 128, 128, 128)
  ]

  initializer = tf.random_normal_initializer(0., 0.02)
  last = tf.keras.layers.Conv2DTranspose(out_channels, 4,
                                         strides=2,
                                         padding='same',
                                         kernel_initializer=initializer,
                                         activation='tanh')  # (bs, 256, 256, 3)

  x = inputs

  # Downsampling through the model
  skips = []
  for down in down_stack:
    x = down(x)
    skips.append(x)

  skips = reversed(skips[:-1])

  # Upsampling and establishing the skip connections
  for up, skip in zip(up_stack, skips):
    # print(up)
    # print(skip)
    x = up(x)
    x = tf.keras.layers.Concatenate()([x, skip])

  x = last(x)

  return tf.keras.Model(inputs=inputs, outputs=x)


def main():
    
    os.environ["CUDA_VISIBLE_DEVICES"] = "14"
    gpus = tf.config.experimental.list_physical_devices('GPU')

    tf.config.experimental.set_memory_growth(gpus[0] , True)

    patch_dim = (64, 64)
    # random_state_patches, num_patches = 15, 10
    fundus_channels, oct_channels = 3, 4

    Generator(in_channels=fundus_channels, patch_dim=patch_dim, out_channels=oct_channels)

if __name__ == '__main__':
    main()