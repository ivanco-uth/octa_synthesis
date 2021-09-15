import os
import re
import sys
import math
import time
import numpy as np
from shutil import rmtree
from numpy.lib.arraypad import pad
from math import floor, ceil
import pandas as pd
from pandas.core import algorithms
from pandas.io.formats.format import ExtensionArrayFormatter, return_docstring
from scipy.sparse import data
from skimage import io
from skimage.transform import resize, rotate
from matplotlib import pyplot as plt
import tensorflow as tf
from sklearn.feature_extraction.image import extract_patches_2d
# from scipy.ndimage import rotate 
from skimage.segmentation import watershed, random_walker
from skimage.filters import sobel
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu
from tensorflow.python.ops.gen_batch_ops import batch
from dataset_load_gen_mod import load_data_cases, load_image_train, break_img_to_patches, put_img_back_from_patches, load_image_train_single
from pix_generator import Generator
from pix_discriminator import Discriminator
from IPython import display
from skimage.io import imread, imsave
from get_quality_cases import get_quality_cases_list

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
gpus = tf.config.experimental.list_physical_devices('GPU')

tf.config.experimental.set_memory_growth(gpus[0] , True)


import datetime
log_dir="logs/"

summary_writer = tf.summary.create_file_writer(
  log_dir + "fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))


# function to extract patches from ragged tensor...
def map_func(in_fundus, in_oct):
    entry_x, entry_y = in_fundus.to_tensor(), in_oct.to_tensor()
    print("x: ", entry_x.shape)
    print("y: ", entry_y.shape)

    joint_entries = tf.concat([entry_x, entry_y], axis=-1)
            # entry_x, entry_y = tf.expand_dims(entry_x), tf.expand_dims(entry_y)
    joint_crop = tf.image.random_crop(value=joint_entries, size=(512, 512, 4))
    joint_crop = tf.image.random_flip_left_right(image = joint_crop)
    joint_crop = tf.image.random_flip_up_down(image = joint_crop)


    crop_fundus, crop_oct = joint_crop[:, :, 0:3], joint_crop[:, : , 3]

    return crop_fundus, crop_oct


def generator_loss(disc_generated_output, gen_output, target, loss_object, LAMBDA):
  gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)

  # mean absolute error

  target = tf.cast(target, dtype=tf.float32)
  target = tf.expand_dims(target, axis=-1)


  l1_loss = tf.reduce_mean(tf.abs(target - gen_output))

  total_gen_loss = gan_loss + (LAMBDA * l1_loss)

  return total_gen_loss, gan_loss, l1_loss


def discriminator_loss(disc_real_output, disc_generated_output, loss_object):
  real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)

  generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)

  total_disc_loss = real_loss + generated_loss

  return total_disc_loss




@tf.function
def train_step(input_image, target, epoch, generator, discriminator, generator_opt, discriminator_opt, loss_object, LAMBDA):
  with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
    gen_output = generator(input_image, training=True)

    disc_real_output = discriminator([input_image, target], training=True)
    disc_generated_output = discriminator([input_image, gen_output], training=True)

    gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(disc_generated_output, gen_output, target, loss_object, LAMBDA)
    disc_loss = discriminator_loss(disc_real_output, disc_generated_output, loss_object)

  generator_gradients = gen_tape.gradient(gen_total_loss,
                                          generator.trainable_variables)
  discriminator_gradients = disc_tape.gradient(disc_loss,
                                               discriminator.trainable_variables)

  generator_opt.apply_gradients(zip(generator_gradients,
                                          generator.trainable_variables))
  discriminator_opt.apply_gradients(zip(discriminator_gradients,
                                              discriminator.trainable_variables))

  
  return gen_total_loss, disc_loss


def generate_images(model, test_input, tar):
  prediction = model(test_input, training=True)

  plt.figure(figsize=(15, 15))

  display_list = [test_input[0], tar[0], prediction[0]]
  title = ['Input Image', 'Ground Truth', 'Predicted Image']

  for i in range(3):
    plt.subplot(1, 3, i+1)
    plt.title(title[i])
    # getting the pixel values between [0, 1] to plot it.
    plt.imshow(display_list[i] * 0.5 + 0.5)
    plt.axis('off')
  plt.show()


def evaluate_whole_fundus(model, image_case, data_fold, patch_size, batch_size, epoch):

  fundus_sample_x = imread("data/nasa_data/sub-00000{0}/ses-01/fundus/000060__78_Image_OS.jpg".format(image_case))

  shape_fundus_x, shape_fundus_y = fundus_sample_x.shape[0], fundus_sample_x.shape[1]

  pad_init_x, pad_init_y = fundus_sample_x.shape[0]//4, fundus_sample_x.shape[1]//4

  # fundus_sample_x = np.pad(fundus_sample_x, pad_width= ((413 , 413), (413 , 413), (0, 0)), constant_values = 0)

  fundus_sample_x = np.pad(fundus_sample_x, pad_width= ((pad_init_x, pad_init_x), (pad_init_y, pad_init_y), (0, 0)), constant_values = 0)
      
  fundus_sample_x = rotate(fundus_sample_x, angle=0, resize=False, preserve_range=True)

  fundus_sample_x = fundus_sample_x/255

  fundus_sample_x = (fundus_sample_x / 0.5) - 1

  

  patch_dataset, num_patches = break_img_to_patches(img_input=fundus_sample_x,
       patch_size=patch_size, return_dataset=True, stride=2)

  print("Number of patches: {0}".format(num_patches))

  print(patch_dataset)

  # patch_dataset = patch_dataset.map(load_image_train,
  #                                   num_parallel_calls=tf.data.experimental.AUTOTUNE)
  # print(patch_dataset)
  patch_dataset.batch(batch_size=batch_size)

  print(patch_dataset)

  # ckpt = tf.train.Checkpoint(generator=generator, generator_optimizer=generator_optimizer)
  # ckpt.restore(tf.train.latest_checkpoint(checkpoint_dir)).expect_partial()

  patch_array = np.zeros(shape=(num_patches, patch_size[0], patch_size[1], 4))

  for patch_idx, (patch_fundus) in enumerate(patch_dataset):
    # print("Here"
    patch_fundus = tf.expand_dims(patch_fundus, axis=0)
    pred_patch = model(patch_fundus, training=True)
    patch_array[patch_idx, :, :, :] = pred_patch



  rec_img = put_img_back_from_patches(patches=patch_array , out_img_resolution=fundus_sample_x.shape, stride=2)

  rec_img = rec_img[pad_init_x:pad_init_x+shape_fundus_x, pad_init_y:pad_init_y+shape_fundus_y]

  # imsave(fname="pix_out.png", arr=rec_img)

  # rec_img = np.average(rec_img, axis=0)

  # print(rec_img.shape)

  # # rec_img[bg_mask == 1] = 0

  rec_img = rec_img * 0.5 + 0.5

  # plt.figure()
  # plt.imshow(rec_img)
  # plt.show()

  rec_img = np.expand_dims(rec_img, axis=0)

  with summary_writer.as_default():
    tf.summary.image('Whole Fundus', rec_img, step=epoch)




# @tf.function
def generate_fundus_oct_pair(model, image_case, data_fold, patch_size, batch_size, epoch):

  print("Generating Image")
  eye_list = ["OS", "OD"]
  fundus_img_path = "{0}/sub-00000{1}/{2}/{3}".format(data_fold, image_case, "OS", "fundus_disc.npy")
  oct_img_path = "{0}/sub-00000{1}/{2}/{3}".format(data_fold, image_case, "OS", "oct_disc.npy")

  disc_sample_x = np.load(file=fundus_img_path)
  disc_sample_y = np.load(file=oct_img_path)

  shape_fundus_x, shape_fundus_y = disc_sample_x.shape[0], disc_sample_x.shape[1]

  # disc_sample_x = disc_sample_x[0:512, 0:512]

  # disc_sample_y = disc_sample_y[0:512, 0:512]


  # fit_match_x, fit_match_y = int(ceil(shape_fundus_x / patch_size[0])), int(ceil(shape_fundus_y /  patch_size[1]))

  # print(fit_match_x, fit_match_y)

  # pad_match_x, pad_match_y = (fit_match_x * patch_size[0]) - shape_fundus_x, (fit_match_y * patch_size[1]) - shape_fundus_y

  # print(pad_match_x, pad_match_y)

  # disc_sample_x = np.pad(disc_sample_x, pad_width= ((0 , pad_match_x), (0 , pad_match_y), (0, 0)), constant_values = -1)
  # disc_sample_y = np.pad(disc_sample_y, pad_width= ((0 , pad_match_x), (0 , pad_match_y), (0, 0)), constant_values = -1)

  patch_dataset, num_patches = break_img_to_patches(img_input=disc_sample_x,
   img_label=disc_sample_y, patch_size=patch_size, return_dataset=True, stride=2)
  
  
  patch_dataset = patch_dataset.map(load_image_train,
                                    num_parallel_calls=tf.data.experimental.AUTOTUNE)
  # print(patch_dataset)
  patch_dataset.batch(batch_size=batch_size)

  # print(patch_dataset)

  

  cnt = 0 

  # print(tf.size(patch_dataset))

  # sys.exit()

  patch_predicts = np.zeros((num_patches, patch_size[0], patch_size[1], 1))

  print("Total patches: {0}".format(num_patches))

  for idx, (patch_fundus, patch_oct) in enumerate(patch_dataset):
    patch_fundus = tf.expand_dims(patch_fundus, axis=0)
    patch_prediction = model(patch_fundus, training=True)
    patch_predicts[idx, :, :] = patch_prediction.numpy()

  print(patch_predicts.shape)

  rec_img = put_img_back_from_patches(patches=patch_predicts , out_img_resolution=disc_sample_x.shape, stride=2)

  print("Shape reconstructed array: {0}".format( rec_img.shape ))
    
  # rec_img = np.average(rec_img, axis=0)

  # rec_img = rec_img[0, ...]

  disc_sample_x = disc_sample_x[0:shape_fundus_x, 0:shape_fundus_y]
  disc_sample_y = disc_sample_y[0:shape_fundus_x, 0:shape_fundus_y]

  rec_img = rec_img[0:shape_fundus_x, 0:shape_fundus_y]

  # plt.figure(figsize=(15, 15))

  # display_list = [disc_sample_x, disc_sample_y, rec_img]
  # title = ['Input Image', 'Ground Truth', 'Predicted Image']


  # for i in range(3):
  #   plt.subplot(1, 3, i+1)
  #   plt.title(title[i])
  #   # getting the pixel values between [0, 1] to plot it.
  #   plt.imshow(display_list[i] * 0.5 + 0.5)
  #   plt.axis('off')
  # plt.show()

  # rec_img = np.reshape(rec_img, (-1, rec_img.shape[0], rec_img.shape[1], rec_img.shape[-1]))

  # print("Unique Vals: {0}".format(np.unique(rec_img)))

  print(disc_sample_y.shape)
  print(rec_img.shape)

  disc_sample_x = np.expand_dims(disc_sample_x* 0.5 + 0.5, axis=0)
  disc_sample_y = np.expand_dims(np.expand_dims(disc_sample_y* 0.5 + 0.5, axis=0), axis=-1)
  rec_img = np.expand_dims(rec_img* 0.5 + 0.5, axis=0)

  # rec_img = rec_img[:, :, :, 0:3]

  with summary_writer.as_default():
    tf.summary.image('Fundus', disc_sample_x, step=epoch)
    tf.summary.image('Target OCT', disc_sample_y, step=epoch)
    tf.summary.image('Predicted OCT', rec_img, step=epoch)

  # sys.exit()

  # plt.figure(figsize=(15, 15))

  # display_list = [test_input[0], tar[0], prediction[0]]
  # title = ['Input Image', 'Ground Truth', 'Predicted Image']

  


  # disc_sample_x = np.expand_dims(disc_sample_x, axis=1)
  # disc_sample_y = np.expand_dims(disc_sample_y, axis=1)



  # check_dataset = tf.data.Dataset.from_tensor_slices((disc_sample_x, disc_sample_y))
  # check_dataset = check_dataset.map(load_image_train,
                                    # num_parallel_calls=tf.data.experimental.AUTOTUNE)
  
  # for example_input, example_target in check_dataset.take(1):

  # prediction = model(test_input, training=True)


def main():

    train_chk = True
    eye_list = ["OS", "OD"]
    # data_fold = "data/formatted_nasa_fundus_oct"
    data_fold = "data/align_09_12"
    
    patch_dim = (512, 512)
    random_state_patches, num_patches = 15, 10
    fundus_channels, oct_channels = 3, 1
    batch_size = 32

    train_set = list(range(1, 31)) 
    test_set = list(range(51, 53))

    all_sets = list(range(1, 122))

    file_path_xlx = "NASAStroke_DATA_2021-09-02_1416.xlsx"

    q_case_list = get_quality_cases_list(path_to_file=file_path_xlx, lowest_quality="0", case_range=all_sets)


    LAMBDA = 100
    epochs = 2000

    loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    if train_chk == True:

      # print("Num images in quality list: {0}".format(len(q_case_list)))

      
      
      train_disc_x, train_disc_y, train_macula_x, train_macula_y = load_data_cases(data_fold = data_fold, list_cases= train_set, eye_list= eye_list, patch_dim= patch_dim, random_state_patches = random_state_patches
  , num_patches = num_patches, fundus_channels = fundus_channels, oct_channels = oct_channels, list_quality_cases = q_case_list)

      print(train_disc_y[0].shape)
      # sys.exit()

      test_disc_x, test_disc_y, test_macula_x, test_macula_y = load_data_cases(data_fold = data_fold, list_cases= test_set, eye_list= eye_list, patch_dim= patch_dim, random_state_patches = random_state_patches
  , num_patches = num_patches, fundus_channels = fundus_channels, oct_channels = oct_channels, list_quality_cases = q_case_list)

      # print(len(train_disc_x))

      train_data_x = train_disc_x + train_macula_x
      train_data_y = train_disc_y + train_macula_y

      # check why there are more images in resulting training data...

      # print("Total images in train dataset: {0}".format(len(train_data_x)))

      # sys.exit()

      train_data_x = tf.ragged.constant(train_data_x)
      train_data_y = tf.ragged.constant(train_data_y)

      test_data_x = tf.ragged.constant(test_disc_x)
      test_data_y = tf.ragged.constant(test_disc_y)

      # train_macula_x = tf.ragged.constant(train_macula_x)
      # train_macula_y = tf.ragged.constant(train_macula_y)

      # print(train_disc_y.shape)
      # print(train_macula_x.shape)
      # print(train_macula_y.shape)

      # train_data_x = np.concatenate((train_disc_x, train_macula_x), axis=0)
      # train_data_y = np.concatenate((train_disc_y, train_macula_y), axis=0)


      # sys.exit()

      # train_disc_x = np.expand_dims(train_disc_x, axis=1)
      # train_disc_y = np.expand_dims(train_disc_y, axis=1)

      # test_disc_x = np.expand_dims(test_disc_x, axis=1)
      # test_disc_y = np.expand_dims(test_disc_y, axis=1)

      # print(train_disc_x.shape)
      # print(train_disc_y.shape)

      # plt.imshow(test_disc_x[0][0])
      # plt.figure()
      # plt.imshow(test_disc_y[0][0])
      # plt.show()
      # sys.exit()


      train_dataset = tf.data.Dataset.from_tensor_slices((train_data_x, train_data_y))

      # sys.exit()

      # train_dataset = tf.data.Dataset.from_tensor_slices((train_disc_x, train_disc_y))
      # train_dataset = train_dataset.map(load_image_train,
      #                                 num_parallel_calls=tf.data.experimental.AUTOTUNE)

      # train_dataset.shuffle(buffer_size=train_disc_x.shape[0])
      # train_dataset.batch(batch_size=batch_size)

      # print(train_dataset)

      # sys.exit()

      test_dataset = tf.data.Dataset.from_tensor_slices((test_data_x, test_data_y))
      # test_dataset = test_dataset.map(load_image_train,
      #                                 num_parallel_calls=tf.data.experimental.AUTOTUNE)

      # test_dataset.shuffle(buffer_size=test_disc_x.shape[0]*2, seed=2)
      # test_dataset.batch(batch_size=batch_size)


      generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
      discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

      generator = Generator(in_channels=fundus_channels, patch_dim=patch_dim, out_channels=oct_channels)
      discriminator = Discriminator(in_channels =fundus_channels, out_channels=oct_channels, patch_dim=patch_dim)
      

      generator.summary()
      discriminator.summary()

      # sys.exit()



      checkpoint_dir = './training_checkpoints'
      checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
      checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                      discriminator_optimizer=discriminator_optimizer,
                                      generator=generator,
                                      discriminator=discriminator)


      # for cnt , (img_fundus, img_oct) in enumerate(test_dataset.take(2)):
      #   # plt.imshow(img_fundus[0] * 0.5 + 0.5)
      #   print(cnt)
      #   if cnt == 0:
      #     plt.figure()
      #     plt.imshow(img_fundus[0]* 0.5 + 0.5)
      #     plt.figure()
      #     plt.imshow(img_oct[0]* 0.5 + 0.5)
      #     plt.show()

      # sys.exit()

      img_case = "60"

      loss_check = 1000

      

      for epoch in range(epochs):
        # start = time.time()

        epoch_loss_avg_gen = tf.keras.metrics.Mean()
        epoch_loss_avg_discr = tf.keras.metrics.Mean()
          
        
        epoch_train_dataset = train_dataset.repeat(20)
        epoch_train_dataset = epoch_train_dataset.map(map_func,
                                      num_parallel_calls=tf.data.experimental.AUTOTUNE)

        

        epoch_train_dataset = epoch_train_dataset.batch(batch_size).prefetch(4)


        # test_dataset.shuffle(buffer_size=test_disc_x.shape[0])

        display.clear_output(wait=True)

        generate_fundus_oct_pair(model=generator, image_case= img_case, 
        data_fold = data_fold, patch_size=patch_dim, batch_size=batch_size, epoch=epoch)

        # if (epoch + 1) % 10 == 0:
        #   evaluate_whole_fundus(model=generator, image_case = img_case, data_fold=data_fold, patch_size=patch_dim, batch_size=batch_size, epoch=epoch)


        # sys.exit()

        # for example_input, example_target in test_dataset.take(1):
        #   generate_images(generator, example_input, example_target)
        print("Epoch: ", epoch)

        # Train
        for n, (input_image, target) in epoch_train_dataset.enumerate():
          print('.', end='')
          # print("Type InImage: {0}".format(input_image.dtype))
          # input_image = input_image.to_tensor()
          # target = target.to_tensor()


          if (n+1) % 100 == 0:
            print()
          gen_adv_loss, disc_loss = train_step(input_image, target, epoch, generator, discriminator
          , generator_opt=generator_optimizer, discriminator_opt=discriminator_optimizer, loss_object=loss_object, LAMBDA=LAMBDA)

        
          epoch_loss_avg_gen.update_state(gen_adv_loss)
          epoch_loss_avg_discr.update_state(disc_loss)

        print()

        if epoch_loss_avg_gen.result() < loss_check:
            print("Saved Gen Weights")
            checkpoint.save(file_prefix=checkpoint_prefix)
            loss_check = epoch_loss_avg_gen.result()

        with summary_writer.as_default():
          tf.summary.scalar('gen_total_loss', epoch_loss_avg_gen.result(), step=epoch)
          # tf.summary.scalar('gen_gan_loss', gen_gan_loss, step=epoch)
          # tf.summary.scalar('gen_l1_loss', gen_l1_loss, step=epoch)
          tf.summary.scalar('disc_loss', epoch_loss_avg_discr.result(), step=epoch)

        # saving (checkpoint) the model every 20 epochs
        # if (epoch + 1) % 20 == 0:
        #   checkpoint.save(file_prefix=checkpoint_prefix)

        # print ('Time taken for epoch {} is {} sec\n'.format(epoch + 1,
        #                                                     time.time()-start))
      # checkpoint.save(file_prefix=checkpoint_prefix)

    else:

      # patch_dim = (512, 512)
      image_case = "60"
      stride_val = 8
      # batch_size = 16
      # data_fold = "data/nasa_fundus_oct"
      # fundus_channels, oct_channels = 3, 4

      generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
      generator = Generator(in_channels=fundus_channels, patch_dim=patch_dim, out_channels=oct_channels)

      fundus_img_path = "{0}/sub-00000{1}/{2}/{3}".format(data_fold, image_case, "OS", "fundus_disc.npy")
      oct_img_path = "{0}/sub-00000{1}/{2}/{3}".format(data_fold, image_case, "OS", "oct_disc.npy")


      checkpoint_dir = './training_checkpoints'
      checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")


      disc_sample_x = np.load(file=fundus_img_path)
      oct_sample_y = np.load(file=oct_img_path)

      fundus_sample_x = imread("data/nasa_05_23_21/sub-0000044/ses-01/fundus/OD.jpg")

      fundus_gray = rgb2gray(fundus_sample_x)

      shape_fundus_x, shape_fundus_y = fundus_sample_x.shape[0], fundus_sample_x.shape[1]
      
      pad_init_x, pad_init_y = fundus_sample_x.shape[0]//4, fundus_sample_x.shape[1]//4
    
      # creates mask to remove background based on 0 values in fundus image
      bg_mask = np.zeros((shape_fundus_x, shape_fundus_y))
      bg_mask[fundus_gray == 0] = 1
      
      fundus_sample_x = np.pad(fundus_sample_x, pad_width= ((pad_init_x, pad_init_x), (pad_init_y, pad_init_y), (0, 0)), constant_values = 0)
      
      fundus_sample_x = rotate(fundus_sample_x, angle=0, resize=False, preserve_range=True)

      fundus_sample_x = fundus_sample_x/255

      fundus_sample_x = (fundus_sample_x / 0.5) - 1

      
      print(shape_fundus_x, shape_fundus_y)

      fit_match_x, fit_match_y = int(ceil(shape_fundus_x / patch_dim[0])), int(ceil(shape_fundus_y /  patch_dim[1]))

      print(fit_match_x, fit_match_y)

      pad_match_x, pad_match_y = (fit_match_x * patch_dim[0]) - shape_fundus_x, (fit_match_y * patch_dim[1]) - shape_fundus_y

      print(pad_match_x, pad_match_y)

      fundus_sample_x = np.pad(fundus_sample_x, pad_width= ((0 , pad_match_x), (0 , pad_match_y), (0, 0)), constant_values = 0)

      print(fundus_sample_x.shape)
      # sys.exit()

      # 826 1753 541 1244

      # fundus_sample_x = fundus_sample_x[826:1753, 541:1244]

      # fundus_sample_x = np.load("data/formatted_nasa_fundus_oct/sub-0000053/OS/fundus_disc.npy")

      # print(fundus_sample_x.shape)

      # sys.exit()

      # fundus_sample_x_gray = rgb2gray(fundus_sample_x)

      # bg_mask = np.zeros((fundus_sample_x_gray.shape[0], fundus_sample_x_gray.shape[1]))
      # bg_mask[fundus_sample_x_gray == 0] = 1

      # fundus_sample_x[bg_mask == 1] = 0


      # disc_sample_y = np.zeros((fundus_sample_x.shape[0], fundus_sample_x.shape[1], 4))
      # eye_sample_y = imread()


      

      # print(np.unique(disc_sample_x))
      # print(np.unique(eye_sample_x))

      # sys.exit()

      # fundus_img_pad = np.pad(fundus_sample_x, pad_width= ((fundus_sample_x.shape[0]//8 , fundus_sample_x.shape[0]//8), 
      # (fundus_sample_x.shape[0]//8 , fundus_sample_x.shape[0]//8), (0, 0)), constant_values = 0)
        
      # fundus_sample_x = rotate(fundus_img_pad, angle=0, resize=False, preserve_range=False)

      # print(fundus_sample_x.shape)



      # add steps division for patch extraction
      patch_dataset, num_patches = break_img_to_patches(img_input=fundus_sample_x,
       patch_size=patch_dim, return_dataset=True, stride=stride_val)

      print("Number of patches: {0}".format(num_patches))

      print(patch_dataset)

      patch_dataset = patch_dataset.map(load_image_train_single,
                                        num_parallel_calls=tf.data.experimental.AUTOTUNE)
      # print(patch_dataset)
      patch_dataset.batch(batch_size=batch_size)

      print(patch_dataset)

      ckpt = tf.train.Checkpoint(generator=generator, generator_optimizer=generator_optimizer)
      ckpt.restore(tf.train.latest_checkpoint(checkpoint_dir)).expect_partial()

      patch_array = np.zeros(shape=(num_patches, patch_dim[0], patch_dim[1], oct_channels))

      for patch_idx, (patch_fundus) in enumerate(patch_dataset):
        # print("Here"
        pred_patch = generator(patch_fundus, training=True)
        patch_array[patch_idx, :, :, :] = pred_patch


      print(patch_array.shape)
      print(fundus_sample_x.shape)
      

      rec_output = put_img_back_from_patches(patches=patch_array , out_img_resolution=fundus_sample_x.shape, stride=stride_val)

      # imsave(fname="pix_out.png", arr=rec_img)

      # print(avg_array.shape)

      # for array_idx in range(avg_array.shape[0]):
      #   img_array = avg_array[array_idx, :, :, :] 
      #   img_array = img_array * 0.5 + 0.5
      #   plt.figure()
      #   plt.imshow(img_array)

      # plt.show()

      # rec_output = np.average(avg_array, axis=0)

      rec_output = rec_output[pad_init_x:pad_init_x+shape_fundus_x, pad_init_y:pad_init_y+shape_fundus_y]
      rec_output[bg_mask == 1] = 0

      plt.figure()
      plt.imshow(rec_output* 0.5 + 0.5)
      plt.show()


      # sys.exit()

      # rec_img[bg_mask == 1] = 0

      # rec_img = rec_img * 0.5 + 0.5

      # plt.figure()
      # plt.imshow(rec_img)
      # plt.show()



      rec_output = np.expand_dims(rec_output, axis=0)

      rec_output = rec_output * 0.5 + 0.5

      

      with summary_writer.as_default():
        tf.summary.image('Whole Fundus', rec_output, step=0)



if __name__ == '__main__':
    main()