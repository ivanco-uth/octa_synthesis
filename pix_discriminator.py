# libraries
import os
import tensorflow as tf
from pix_generator import downsample


def Discriminator(in_channels, out_channels, patch_dim):

  """
  Defines the discriminator model for the Pix2Pix framework.

  The discriminator evaluates the authenticity of generated images by 
  distinguishing between real and fake pairs of input and target images. 
  This function implements a PatchGAN-based discriminator, processing 
  image patches instead of the entire image, to model high-frequency details.

  Args:
      in_channels (int): Number of input channels (e.g., RGB or grayscale).
      out_channels (int): Number of target output channels.
      patch_dim (tuple): Spatial dimensions of the input patch (height, width).

  Returns:
      tf.keras.Model: A Keras model representing the discriminator.
  """

  
  # kernel initializer definition
  initializer = tf.random_normal_initializer(0., 0.02)

  # input and target layers in pix2pix discriminator
  inp = tf.keras.layers.Input(shape=[patch_dim[0], patch_dim[1], in_channels], name='input_image')
  tar = tf.keras.layers.Input(shape=[patch_dim[0], patch_dim[1], out_channels], name='target_image')

  # joint input having real and fake samples
  x = tf.keras.layers.concatenate([inp, tar])  # (bs, 256, 256, channels*2)

  # convolutional downsampling
  down1 = downsample(128, 4, False)(x)  # (bs, 128, 128, 64)
  
  # convolutional downsampling
  down2 = downsample(256, 4)(down1)  # (bs, 64, 64, 128)
#   down3 = downsample(256, 4)(down2)  # (bs, 32, 32, 256)

  # zero pad before final convolutional layers
  zero_pad1 = tf.keras.layers.ZeroPadding2D()(down2)  # (bs, 34, 34, 256)
  conv = tf.keras.layers.Conv2D(512, 4, strides=1,
                                kernel_initializer=initializer,
                                use_bias=False)(zero_pad1)  # (bs, 31, 31, 512)

  # batch normalization
  batchnorm1 = tf.keras.layers.BatchNormalization()(conv)

  # leaky rely activation
  leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)

  # zero pad before final convolutional layers
  zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)  # (bs, 33, 33, 512)

  # final convolutional layer
  last = tf.keras.layers.Conv2D(1, 4, strides=1,
                                kernel_initializer=initializer)(zero_pad2)  # (bs, 30, 30, 1)

  return tf.keras.Model(inputs=[inp, tar], outputs=last)


def main():

    # test discriminator initialization
    os.environ["CUDA_VISIBLE_DEVICES"] = "14"
    gpus = tf.config.experimental.list_physical_devices('GPU')

    # sets gpu memory to grow 
    tf.config.experimental.set_memory_growth(gpus[0] , True)

    # dummy patch dimension
    patch_dim = (64, 64)
    
    # dummy channel initialization
    fundus_channels, oct_channels = 3, 4

    # test of discrinator module
    Discriminator(fundus_channels, oct_channels, patch_dim)


if __name__ == '__main__':
    main()