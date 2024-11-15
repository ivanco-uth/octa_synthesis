import os
import tensorflow as tf



def downsample(filters, size, apply_batchnorm=True):
  
  """
  Defines a downsampling layer for a U-Net or similar architecture.

  This function creates a sequential downsampling block with a convolutional 
  layer followed optionally by batch normalization and a Leaky ReLU activation. 
  Downsampling reduces the spatial dimensions while increasing the depth of 
  feature maps, aiding in feature extraction.

  Args:
      filters (int): Number of filters in the convolutional layer.
      size (int): Size of the convolutional kernel.
      apply_batchnorm (bool): Whether to include batch normalization after 
                              the convolutional layer (default: True).

  Returns:
      tf.keras.Sequential: A downsampling layer as a sequential model.
  """

  
  #kernel initializer
  initializer = tf.random_normal_initializer(0., 0.02)

  # convolutional layer definition
  result = tf.keras.Sequential()

  # adds convolutional layer to the model
  result.add(
      tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                             kernel_initializer=initializer, use_bias=False))

  # adds batch normalization layer to the model
  if apply_batchnorm:
    result.add(tf.keras.layers.BatchNormalization())

  # adds leaky relu activation layer to the model
  result.add(tf.keras.layers.LeakyReLU())

  return result



def upsample(filters, size, apply_dropout=False):

  """
  Defines an upsampling layer for a U-Net or similar architecture.

  This function creates a sequential upsampling block with a transposed 
  convolutional layer followed by batch normalization and a ReLU activation. 
  Optionally, dropout can be applied. Upsampling increases the spatial 
  dimensions while reducing the depth of feature maps, aiding in image 
  reconstruction.

  Args:
      filters (int): Number of filters in the transposed convolutional layer.
      size (int): Size of the transposed convolutional kernel.
      apply_dropout (bool): Whether to include a dropout layer after batch 
                            normalization (default: False).

  Returns:
      tf.keras.Sequential: An upsampling layer as a sequential model.
  """


  # initializes kernel to random normal distribution
  initializer = tf.random_normal_initializer(0., 0.02)

  # convolutional layer definition
  result = tf.keras.Sequential()

  # adds convolutional layer to the model
  result.add(
    tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                    padding='same',
                                    kernel_initializer=initializer,
                                    use_bias=False))

  # adds batch normalization layer to the model
  result.add(tf.keras.layers.BatchNormalization())

  # adds dropout layer to the model
  if apply_dropout:
      result.add(tf.keras.layers.Dropout(0.5))

  # adds relu activation layer to the model
  result.add(tf.keras.layers.ReLU())

  return result




def Generator(in_channels, patch_dim, out_channels):

  """
  Defines a U-Net-based generator model for image-to-image translation tasks.
  This function constructs a neural network architecture with symmetrical 
  downsampling and upsampling stacks, incorporating skip connections for 
  preserving spatial information. The model takes an input tensor with 
  `in_channels` and `patch_dim` dimensions and produces an output tensor 
  with `out_channels`, using a series of convolutional layers for feature 
  extraction and reconstruction.

  Args:
      in_channels (int): Number of channels in the input tensor.
      patch_dim (tuple): Dimensions (height, width) of the input tensor.
      out_channels (int): Number of channels in the output tensor.

  Returns:
      tf.keras.Model: A compiled generator model.
  """

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