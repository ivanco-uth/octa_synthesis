import os
import re
import sys
import math
import numpy as np
from shutil import rmtree
from numpy.lib.arraypad import pad
import pandas as pd
from pandas.core import algorithms
from pandas.io.formats.format import ExtensionArrayFormatter
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
from math import ceil, floor
from skimage.io import imread
from tensorflow.python.ops.gen_batch_ops import batch

fundus_disc, fundus_macula = "fundus_disc", "fundus_macula"
oct_disc, oct_macula = "oct_disc", "oct_macula"


def back_forth_patches(img_input, img_label=None, patch_size=None, stride=1, mode=None, global_step=0, batch_size=32, img_state=None, pred_img_patch=None):

    patch_dim_x, patch_dim_y = patch_size[0], patch_size[1]
    img_res_x, img_res_y = img_input.shape[0], img_input.shape[1]
    # steps_x, steps_y =  ceil(stride *img_res_x/(patch_dim_x)),  ceil(stride *img_res_y/(patch_dim_y))

    steps_x = ceil(stride * (img_res_x - patch_dim_x)/(patch_dim_x)) + 1
    steps_y = ceil(stride * (img_res_y - patch_dim_y)/(patch_dim_y)) + 1

    # print("Total steps: ", steps_x, steps_y)
    
    # global_step = 1

    

    # print(y_step)
    # sys.exit()

    patch_num = steps_x * steps_y

    x_step, y_step = 0, 0
    
    end_check = False

    if global_step is not None:
        y_step =  int((global_step * batch_size) / steps_x)

    if global_step is None and mode == "extract":
        global_step = 0
        img_state = np.ones((stride, img_res_x, img_res_y, 1), dtype=np.float32)*-1

    
    if patch_num - (global_step * batch_size) < batch_size:
        num_samples_patch = patch_num - (global_step * batch_size)

    else:
        num_samples_patch = batch_size

    
    
    

    img_x_patch = np.ones((num_samples_patch, patch_dim_x, patch_dim_y, img_input.shape[-1]), dtype=np.float32)*-1

    

    

    
    
    # if global_step > 0:


    if mode == "extract":

        # print(global_step, num_samples_patch)
        
        for step in range(global_step*batch_size, (global_step+1)*num_samples_patch):
            # print("Global Step: {0}".format(step))
            
            x_step = step % steps_x

            # print(x_step, y_step)
            x_init, x_end = x_step*patch_dim_x//stride, int(((x_step/stride) + 1) * patch_dim_x)
            y_init, y_end = y_step*patch_dim_y//stride, int(((y_step/stride) + 1) * patch_dim_y)

            # print("X coords: ", x_init, x_end)
            # print("Y coords: ", y_init, y_end)


            if img_res_x % patch_dim_x != 0 and x_step == steps_x - 1 and img_res_y % patch_dim_y != 0 and y_step == steps_y - 1:
                # print("last position xy")
                img_x_patch[step%batch_size,] = img_input[img_res_x - patch_dim_x:img_res_x, img_res_y - patch_dim_y:img_res_y]
                

            elif img_res_x % patch_dim_x != 0 and x_step == steps_x - 1:
                # print("last position x ")
                img_x_patch[step%batch_size,] = img_input[img_res_x - patch_dim_x :img_res_x, y_init:y_end]
                

            elif img_res_y % patch_dim_y != 0 and y_step == steps_y - 1:
                # print("last position y ")
                # print(img_res_y - patch_dim_y - 1: :img_res_y -1)
                img_x_patch[step%batch_size,] = img_input[x_init:x_end, img_res_y - patch_dim_y:img_res_y]
                

            else:
                img_x_patch[step%batch_size,] = img_input[x_init:x_end, y_init:y_end]

            
            if (step + 1) % steps_x == 0 and step > 0:
                y_step += 1
                
                

        # print(y_step)
        # sys.exit()
        

        return img_x_patch, img_state, end_check, global_step


    if mode == "merge":

        

        patch_dim_x, patch_dim_y = pred_img_patch.shape[1], pred_img_patch.shape[2]

        # print("Patches: ", patch_dim_x, patch_dim_y)

        # steps_x, steps_y = ceil(img_res_x/patch_dim_x), ceil(img_res_y/patch_dim_y)

        steps_x = ceil(stride * (img_res_x - patch_dim_x)/(patch_dim_x)) + 1
        steps_y = ceil(stride * (img_res_y - patch_dim_y)/(patch_dim_y)) + 1

        # print("Steps: ", steps_x, steps_y)


        rec_img = np.zeros((img_res_x, img_res_y, 4)) # value was 4

        # print(rec_img.shape)

        # x_step = 0
        # y_step = 0

        avg_array = img_state # should fix this
        
        # avg_array *= 1000
        # avg_array = avg_array.astype(int)

        # print(avg_array.shape)

        for step, slice in enumerate(pred_img_patch):
            # print(step)
            step = (global_step*batch_size)+step
            # print("Step inside reconstruction: ", step)

            x_step = step % steps_x
            
            # print(x_step, y_step)
            # print(x_step, y_step)


            x_init, x_end = x_step*patch_dim_x//stride, int(((x_step/stride) + 1) * patch_dim_x)
            y_init, y_end = y_step*patch_dim_y//stride, int(((y_step/stride) + 1) * patch_dim_y)

            # print("X coords: ", x_init, x_end)
            # print("Y coords: ", y_init, y_end)

            x_limit_init, x_limit_end = img_res_x - patch_dim_x - 1, img_res_x -1
            y_limit_init, y_limit_end = img_res_y - patch_dim_y - 1, img_res_y - 1


            avg_array_index = step % stride

            # print("Index in AVG ", avg_array_index)

            # print(step % stride)

            # if step % stride == 0:
                # print(x_init, x_end)
                # print(y_init, y_end)

            # print("Now")
            

            # print("Next")
            # if x_step != steps_x - 1:
            #     next_x_init, next_x_end = (x_step + 1) * patch_dim_x//stride, int((((x_step + 1)/stride) + 1) * patch_dim_x)
            #     # print(next_x_init, next_x_end)
            
            # if y_step != steps_y - 1:
            #     next_y_init, next_y_end = (y_step + 1) * patch_dim_y//stride, int((((y_step + 1)/stride) + 1) * patch_dim_y)
                # print(next_y_init, next_y_end)


            if img_res_x % patch_dim_x != 0 and x_step == steps_x - 1 and img_res_y % patch_dim_y != 0 and y_step == steps_y - 1:
                # print("last position xy")
                # print(x_limit_init, x_limit_end)
                # print(y_limit_init, y_limit_end)
                # img_x_patch = img_input[img_res_x - patch_dim_x - 1 :img_res_x -1, img_res_y - patch_dim_y - 1:img_res_y -1]
                # rec_img[img_res_x - patch_dim_x - 1:img_res_x -1, img_res_y - patch_dim_y - 1:img_res_y - 1] = slice
                avg_array[avg_array_index, img_res_x - patch_dim_x:img_res_x, img_res_y - patch_dim_y:img_res_y] = slice

            elif img_res_x % patch_dim_x != 0 and x_step == steps_x - 1:
                # print("last position x ")
                # print(x_limit_init, x_limit_end)
            
                # img_x_patch = img_input[img_res_x - patch_dim_x - 1 :img_res_x -1, y_step*patch_dim_y:(y_step+1)*patch_dim_y]
                # rec_img[img_res_x - patch_dim_x - 1 :img_res_x -1, y_init:y_end] = slice
                avg_array[avg_array_index, img_res_x - patch_dim_x :img_res_x, y_init:y_end] = slice
            
            elif img_res_y % patch_dim_y != 0 and y_step == steps_y - 1:
                # print("last position y ")
                # print(y_limit_init, y_limit_end)
                # img_x_patch = img_input[x_step*patch_dim_x:(x_step+1)*patch_dim_x, img_res_y - patch_dim_y - 1: :img_res_y -1]
                # print(img_res_y - patch_dim_y - 1: :img_res_y -1)
                # rec_img[x_init:x_end, img_res_y - patch_dim_y - 1:img_res_y -1] = slice
                avg_array[avg_array_index, x_init:x_end, img_res_y - patch_dim_y:img_res_y] = slice
            
            else:   
                # img_x_patch = img_input[x_step*patch_dim_x:(x_step+1)*patch_dim_x, y_step*patch_dim_y:(y_step+1)*patch_dim_y]
                # rec_img[x_init:x_end, y_init:y_end] = slice
                # print("within array")
                avg_array[avg_array_index, x_init:x_end, y_init:y_end] = slice
                # print(x_init, x_end)
                # print(y_init, y_end)
            

            
            # print(img_y_patch.shape)
            # patch_x_arr[step, :, :, :] = img_x_patch
            # patch_y_arr[step, :, :, :] = img_y_patch

            if (step +1) % steps_x == 0 and step > 0:
                y_step += 1

        

        global_step += 1

        # print("Before out check: ",patch_num, global_step, batch_size)
        if patch_num < ((global_step) * batch_size):
            # print("Out conditions: ", patch_num, global_step, batch_size)
            end_check = True

        return avg_array, global_step, end_check



        

        # sys.exit()

        


        

    # return set_patch_ds, patch_x_arr, end_check, global_step 
   


# @tf.function
def break_img_to_patches(img_input, img_label=None, patch_size=None, return_dataset=None, stride=1):

    patch_dim_x, patch_dim_y = patch_size[0], patch_size[1]
    img_res_x, img_res_y = img_input.shape[0], img_input.shape[1]
    # steps_x, steps_y =  ceil(stride *img_res_x/(patch_dim_x)),  ceil(stride *img_res_y/(patch_dim_y))

    steps_x = ceil(stride * (img_res_x - patch_dim_x)/(patch_dim_x)) + 1
    steps_y = ceil(stride * (img_res_y - patch_dim_y)/(patch_dim_y)) + 1

    # print(steps_x, steps_y)
    
    patch_num = steps_x * steps_y

    patch_x_arr = np.zeros((patch_num, patch_dim_x, patch_dim_y, img_input.shape[-1]), dtype=np.float32)
    patch_y_arr = None

    if img_label is not None:
        patch_y_arr = np.zeros((patch_num, patch_dim_x, patch_dim_y, img_label.shape[-1]))


    # print("Shape Input: {0}".format(img_input.shape))
    # print(patch_dim_x, img_res_x)
    # print(img_res_x % patch_dim_x)
    # print(img_res_y % patch_dim_y)
    x_step = 0
    y_step = 0

    # plt.figure()
    # plt.imshow(img_input)
    # plt.show()

    # sys.exit()

    # print("Num steps: {0}".format(patch_num))


    # sys.exit()

    # plt.figure()
    for step in range(patch_num):
        # print(step)
        

        x_step = step % steps_x
        
        # print(x_step, y_step)
        x_init, x_end = x_step*patch_dim_x//stride, int(((x_step/stride) + 1) * patch_dim_x)
        y_init, y_end = y_step*patch_dim_y//stride, int(((y_step/stride) + 1) * patch_dim_y)

        # print(x_init, x_end)
        # print(y_init, y_end)

        # print(x_step, y_step)

        if img_res_x % patch_dim_x != 0 and x_step == steps_x - 1 and img_res_y % patch_dim_y != 0 and y_step == steps_y - 1:
            # print("last position xy")
            img_x_patch = img_input[img_res_x - patch_dim_x:img_res_x, img_res_y - patch_dim_y:img_res_y]

            if img_label is not None:
                img_y_patch = img_label[img_res_x - patch_dim_x :img_res_x, img_res_y - patch_dim_y:img_res_y]
            

        elif img_res_x % patch_dim_x != 0 and x_step == steps_x - 1:
            # print("last position x ")
            img_x_patch = img_input[img_res_x - patch_dim_x :img_res_x, y_init:y_end]
            
            if img_label is not None:
                img_y_patch = img_label[img_res_x - patch_dim_x  :img_res_x, y_init:y_end]
        
        elif img_res_y % patch_dim_y != 0 and y_step == steps_y - 1:
            # print("last position y ")
            # print(img_res_y - patch_dim_y - 1: :img_res_y -1)
            img_x_patch = img_input[x_init:x_end, img_res_y - patch_dim_y:img_res_y]
            
            if img_label is not None:
                img_y_patch = img_label[x_init:x_end, img_res_y - patch_dim_y : img_res_y ]

        else:
            # print(x_init, x_end)
            # print(y_init, y_end)
            # print(img_input.shape)
            img_x_patch = img_input[x_init:x_end, y_init:y_end]
            # plt.imshow(img_x_patch)
            # plt.show()
            # print(img_x_patch.shape)
            
            if img_label is not None:
                img_y_patch = img_label[x_init:x_end, y_init:y_end]

        
        # plt.subplot(1, 2, 1)
        # plt.imshow(img_x_patch)
        # plt.subplot(1, 2, 2)
        # plt.imshow(img_y_patch)
        # plt.show()

        # print(img_input.shape)
        # print(img_x_patch.shape)
        # print(img_y_patch.shape)
        # print(patch_x_arr.shape)
        patch_x_arr[step, :, :, :] = img_x_patch

        if img_label is not None:
            patch_y_arr[step, :, :, :] = img_y_patch

        # sys.exit()

        if (step +1) % steps_x == 0 and step > 0:
            y_step += 1

    # comment here  
    # patch_x_arr = np.expand_dims(patch_x_arr, axis=1)
    # patch_y_arr = np.expand_dims(patch_y_arr, axis=1)


    # print(patch_x_arr.shape)

    # comment here
    # if img_label is not None:
    #     patch_y_arr = np.expand_dims(patch_y_arr, axis=1)

    # print("Got Here")
    if return_dataset:

        if img_label is not None:
            check_dataset = tf.data.Dataset.from_tensor_slices((patch_x_arr, patch_y_arr))

        else:
            # print("Here3")
            # print(patch_x_arr.shape)
            # patch_y_arr = np.ones(shape=patch_x_arr.shape)
            # print(patch_y_arr.shape)
            check_dataset = tf.data.Dataset.from_tensor_slices((patch_x_arr))
        # print(check_dataset)

        # print("Here4")
        
        return check_dataset, patch_num

    else:

        return patch_x_arr, patch_y_arr



def put_img_back_from_patches(patches, out_img_resolution, stride=1):

    # print(patches.shape)

    patch_dim_x, patch_dim_y = patches.shape[1], patches.shape[2]

    # print("Patches: ", patch_dim_x, patch_dim_y)

    img_res_x, img_res_y = out_img_resolution[0], out_img_resolution[1]
    # steps_x, steps_y = ceil(img_res_x/patch_dim_x), ceil(img_res_y/patch_dim_y)

    steps_x = ceil(stride * (img_res_x - patch_dim_x)/(patch_dim_x)) + 1
    steps_y = ceil(stride * (img_res_y - patch_dim_y)/(patch_dim_y)) + 1

    # print("Steps: ", steps_x, steps_y)


    rec_img = np.zeros((img_res_x, img_res_y, 4)) # value was 4

    print(rec_img.shape)

    x_step = 0
    y_step = 0

    avg_array = np.zeros((stride + 1, img_res_x, img_res_y, 4)) # should fix this
    avg_array += 1
    avg_array *= 1000
    # avg_array = avg_array.astype(int)

    # print(avg_array.shape)

    for step, slice in enumerate(patches):
        # print(step)

        x_step = step % steps_x
        
        # print(x_step, y_step)
        # print(x_step, y_step)


        x_init, x_end = x_step*patch_dim_x//stride, int(((x_step/stride) + 1) * patch_dim_x)
        y_init, y_end = y_step*patch_dim_y//stride, int(((y_step/stride) + 1) * patch_dim_y)

        x_limit_init, x_limit_end = img_res_x - patch_dim_x - 1, img_res_x -1
        y_limit_init, y_limit_end = img_res_y - patch_dim_y - 1, img_res_y - 1


        avg_array_index = step % stride

        print(step % stride)

        # if step % stride == 0:
            # print(x_init, x_end)
            # print(y_init, y_end)

        # print("Now")
        

        # print("Next")
        # if x_step != steps_x - 1:
        #     next_x_init, next_x_end = (x_step + 1) * patch_dim_x//stride, int((((x_step + 1)/stride) + 1) * patch_dim_x)
        #     # print(next_x_init, next_x_end)
        
        # if y_step != steps_y - 1:
        #     next_y_init, next_y_end = (y_step + 1) * patch_dim_y//stride, int((((y_step + 1)/stride) + 1) * patch_dim_y)
            # print(next_y_init, next_y_end)


        if img_res_x % patch_dim_x != 0 and x_step == steps_x - 1 and img_res_y % patch_dim_y != 0 and y_step == steps_y - 1:
            # print("last position xy")
            # print(x_limit_init, x_limit_end)
            # print(y_limit_init, y_limit_end)
            # img_x_patch = img_input[img_res_x - patch_dim_x - 1 :img_res_x -1, img_res_y - patch_dim_y - 1:img_res_y -1]
            # rec_img[img_res_x - patch_dim_x - 1:img_res_x -1, img_res_y - patch_dim_y - 1:img_res_y - 1] = slice
            avg_array[stride, img_res_x - patch_dim_x:img_res_x, img_res_y - patch_dim_y:img_res_y] = slice

        elif img_res_x % patch_dim_x != 0 and x_step == steps_x - 1:
            # print("last position x ")
            # print(x_limit_init, x_limit_end)
        
            # img_x_patch = img_input[img_res_x - patch_dim_x - 1 :img_res_x -1, y_step*patch_dim_y:(y_step+1)*patch_dim_y]
            # rec_img[img_res_x - patch_dim_x - 1 :img_res_x -1, y_init:y_end] = slice
            avg_array[stride, img_res_x - patch_dim_x :img_res_x, y_init:y_end] = slice
        
        elif img_res_y % patch_dim_y != 0 and y_step == steps_y - 1:
            # print("last position y ")
            # print(y_limit_init, y_limit_end)
            # img_x_patch = img_input[x_step*patch_dim_x:(x_step+1)*patch_dim_x, img_res_y - patch_dim_y - 1: :img_res_y -1]
            # print(img_res_y - patch_dim_y - 1: :img_res_y -1)
            # rec_img[x_init:x_end, img_res_y - patch_dim_y - 1:img_res_y -1] = slice
            avg_array[stride, x_init:x_end, img_res_y - patch_dim_y:img_res_y] = slice
        
        else:   
            # img_x_patch = img_input[x_step*patch_dim_x:(x_step+1)*patch_dim_x, y_step*patch_dim_y:(y_step+1)*patch_dim_y]
            # rec_img[x_init:x_end, y_init:y_end] = slice
            # print("within array")
            avg_array[avg_array_index, x_init:x_end, y_init:y_end] = slice
            # print(x_init, x_end)
            # print(y_init, y_end)
        

        
        # print(img_y_patch.shape)
        # patch_x_arr[step, :, :, :] = img_x_patch
        # patch_y_arr[step, :, :, :] = img_y_patch

        if (step +1) % steps_x == 0 and step > 0:
            y_step += 1

    # print("Unique Final Array: {0}".format(np.unique(avg_array)))

    # sys.exit()

        # for arr_idx in range(avg_array.shape[0]):
        #     zero_array = 
        #     if arr_idx > 0:
        #         current_arr = avg_array[arr_idx, ...]
                
        #         current_arr[avg_array[0, ...] == 5] = avg_array[0, ...]
        #         avg_array[arr_idx, ...] = current_arr

    # weight_arr = np.where(avg_array == 1000, 0, 1)
    # # weight_arr = np.sum(avg_array, axis = 0)

    # print(weight_arr.shape)

    # avg_array = np.average(avg_array, weights=weight_arr, axis=0)
    # print(avg_array.shape)

    # zero_arr = avg_array[0, ...]

    # if avg_array.shape[0] > 1:
    #     for arr_idx in range(avg_array.shape[0]):
    #         current_arr = avg_array[arr_idx, ...]
    #         current_arr = np.where(current_arr == 1000, zero_arr, current_arr) 
    #         avg_array[arr_idx, ...] = current_arr


    # sum_check_arrays = np.zeros((avg_array.shape))
    # for idx in range(avg_array.shape[0]):
    #     current_arr = avg_array[idx, ...]
        
    #     sum_check_arrays[idx, ...] = mean_check_arr
    
    # print(np.unique(avg_array))

    

    


    sum_check_arrays = np.where(avg_array != 1000, 1, 0)

    check_arr = np.sum(sum_check_arrays, axis = 0)

    avg_array = np.where(avg_array == 1000, 0, avg_array)
    sum_arrays = np.sum(avg_array, axis=0)

    out_array = np.divide(sum_arrays, check_arr)

    # print(np.unique(check_arr, return_counts=True))

    # where_arrs = np.where(check_arr == check_arr.min())
    # print(np.unique(where_arrs[0],return_counts=True))
    # print(np.unique(where_arrs[1],return_counts=True))
    # print(where_arrs[1])



    # sys.exit()

    # print(check_arr.shape)

    # print(np.unique(check_arr, return_counts=True) )
    # for idx in range(check_arr.shape[-1]):
    #     plt.figure()
    #     plt.imshow(check_arr[:,:,idx], interpolation=None, cmap="Set1")
    #     # plt.legend(loc='lower right')
    # plt.show()
    




    # sys.exit()

    # for idx in range(avg_array.shape[0]):
    #     plt.figure()
    #     plt.imshow(sum_check_arrays[idx, ...] * avg_array[idx, ...] * 0.5 + 0.5)
        

    # plt.figure()
    # plt.imshow(avg_all_arrays*0.5 + 0.5)

    # plt.show()

    # print(np.unique(np.sum(sum_check_arrays, axis=0)))

    # avg_all_arrays = np.sum(sum_check_arrays, axis=0)
    
    # div_check_arr = np.sum(sum_check_arrays, axis=0)

    # print("Unique Vals: {0}".format(np.unique(div_check_arr)))

    # sum_avg_array = np.sum(avg_array, axis=0)

    # avg_all_arrays = np.divide(sum_avg_array, div_check_arr)

    # for idx in range(avg_array.shape[0]):
    #     plt.figure()
    #     plt.imshow(avg_array[idx, ...]*0.5 + 0.5)
        

    # plt.figure()
    # plt.imshow(out_array*0.5 + 0.5)

    # plt.show()

    # sys.exit()



    # avg_array = np.mean(avg_array, axis=0)

    return out_array

    # sys.exit()
            
            # if img_res_x < (x_step+1)*patch_dim_x and img_res_y < (y_step+1)*patch_dim_y:

            #     img_x_patch = img_input[img_res_x - patch_dim_x - 1, img_res_y - patch_dim_y - 1]
            #     img_y_patch = img_label[img_res_x - patch_dim_x - 1, img_res_y - patch_dim_y - 1]

            # elif img_res_x < (x_step+1)*patch_dim_x:

            #     img_x_patch = img_input[img_res_x - patch_dim_x - 1 :img_res_x -1, y_step*patch_dim_y:(y_step+1)*patch_dim_y]
            #     img_y_patch = img_label[img_res_x - patch_dim_x - 1 :img_res_x -1, y_step*patch_dim_y:(y_step+1)*patch_dim_y]

            # elif img_res_y < (y_step+1)*patch_dim_y:

            #     img_x_patch = img_input[x_step*patch_dim_x:(x_step+1)*patch_dim_x, img_res_y - patch_dim_y - 1]
            #     img_y_patch = img_label[x_step*patch_dim_x:(x_step+1)*patch_dim_x, img_res_y - patch_dim_y - 1]

            # else:    

            #     img_x_patch = img_input[x_step*patch_dim_x:(x_step+1)*patch_dim_x, y_step*patch_dim_y:(y_step+1)*patch_dim_y]
            #     img_y_patch = img_label[x_step*patch_dim_x:(x_step+1)*patch_dim_x, y_step*patch_dim_y:(y_step+1)*patch_dim_y]

            # print(img_x_patch.shape)
            # print(img_y_patch.shape)


def normalize(input_image, real_image):
#   input_image = (input_image / 0.5) - 1
#   real_image = (real_image / 0.5) - 1

  return input_image, real_image

def load_image_train(input_image, real_image):
  input_image, real_image = normalize(input_image, real_image)

  return input_image, real_image

def load_image_train_single(input_image):
  
  input_image = tf.cast(input_image, dtype=tf.float32)

  return input_image

# @tf.function
def load_data_cases(data_fold = "", list_cases=[], eye_list= ["OS", "OD"], patch_dim= (128, 128), random_state_patches = 0
, num_patches = 15, fundus_channels = 3, oct_channels = 4):

    disc_array_x, disc_array_y = np.zeros((num_patches, patch_dim[0], patch_dim[1], fundus_channels)), np.zeros((1, patch_dim[0], patch_dim[1], fundus_channels))
    macula_array_x , macula_array_y = np.zeros((num_patches, patch_dim[0], patch_dim[1], oct_channels)), np.zeros((1, patch_dim[0], patch_dim[1], oct_channels))

    img_set = os.listdir(data_fold)

    disc_count, macula_count = 0, 0

    for img_id in img_set:

        if "sub-" not in img_id:
            continue

        img_number = int(img_id.split("-")[-1].lstrip("0"))

        if img_number not in list_cases:
            continue

        # if img_id == "sub-000005" or img_id == "sub-0000013":
        #     continue

        try:

            img_path = "{0}/{1}".format(data_fold, img_id)
            print(img_path)
            eye_dirs = os.listdir(img_path)
            
            for eye_id in eye_dirs:
                eye_path = "{0}/{1}/{2}".format(data_fold, img_id, eye_id)
                
                if eye_id not in eye_list:
                    try:
                        rmtree(eye_path)
                    except:
                        os.remove(eye_path)

                else:
                    file_list = os.listdir(eye_path)

                    for file in file_list:
                        file_path = "{0}/{1}"

                        if fundus_disc in file:

                            disc_sample_x = np.load(file_path.format(eye_path, file))
                            disc_sample_y = np.load(file_path.format(eye_path, oct_disc + ".npy"))

                            # disc_sample_x = resize(disc_sample_x, patch_dim) 
                            # disc_sample_y = resize(disc_sample_y, patch_dim)

                            # print(disc_sample_x.shape)
                            # print(disc_sample_y.shape)

                            # fundus_disc_patch = extract_patches_2d(disc_sample_x, patch_size = patch_dim
                            # , random_state = random_state_patches, max_patches = num_patches)

                            if disc_sample_x.shape[0] < patch_dim[0] or disc_sample_x.shape[1] < patch_dim[1]:
                                continue

                            fundus_disc_patch, oct_disc_patch = break_img_to_patches(img_input=disc_sample_x, img_label=disc_sample_y
                            , patch_size=patch_dim, return_dataset=False, stride=1)

                            # oct_disc_patch = extract_patches_2d(disc_sample_y, patch_size = patch_dim
                            # , random_state = random_state_patches, max_patches = num_patches)

                            # print(fundus_disc_patch.shape)


                            if disc_count == 0:
                                disc_count = 1
                                
                                disc_array_x = fundus_disc_patch
                                disc_array_y = oct_disc_patch


                            else:

                                print(fundus_disc_patch.shape)
                                print(oct_disc_patch.shape)
                                
                                disc_array_x = np.concatenate((disc_array_x, fundus_disc_patch))
                                disc_array_y = np.concatenate((disc_array_y, oct_disc_patch))


                        if fundus_macula in file :
                            
                            macula_sample_x = np.load(file_path.format(eye_path, file))
                            macula_sample_y = np.load(file_path.format(eye_path, oct_macula + ".npy"))

                            # print(macula_sample_x.shape)
                            # print(macula_sample_y.shape)

                            # fundus_macula_patch = extract_patches_2d(macula_sample_x, patch_size = patch_dim
                            # , random_state = random_state_patches, max_patches = num_patches)
                            # oct_macula_patch = extract_patches_2d(macula_sample_y, patch_size = patch_dim
                            # , random_state = random_state_patches, max_patches = num_patches)

                            if macula_sample_x.shape[0] < patch_dim[0] or macula_sample_x.shape[1] < patch_dim[1]:
                                continue


                            fundus_macula_patch, oct_macula_patch = break_img_to_patches(img_input=macula_sample_x, img_label=macula_sample_y
                            , patch_size=patch_dim, return_dataset=False, stride=1)


                            if macula_count == 0:
                                macula_count = 1

                                macula_array_x = fundus_macula_patch
                                macula_array_y = oct_macula_patch


                            elif fundus_macula in file:
                                
                                macula_array_x = np.concatenate((macula_array_x, fundus_macula_patch))
                                macula_array_y = np.concatenate((macula_array_y, oct_macula_patch))

        except Exception as e:
            print(e)
            # break
            # continue

    

    return disc_array_x, disc_array_y, macula_array_x, macula_array_y



def main():
    fundus_sample_x = imread("data/nasa_05_23_21/sub-0000053/ses-01/fundus/000053__66_Image_OS.jpg")

    img_patches = break_img_to_patches(img_input=fundus_sample_x, img_label=None
    , patch_size=(512, 512), return_dataset=False, stride= 2)


    fundus_patch_array = img_patches[0].astype(int)

    img_out = put_img_back_from_patches(fundus_patch_array, fundus_sample_x.shape, stride=4)

    print(img_out.shape)
    plt.imshow(img_out.astype(int))
    plt.show()



    

    # print(np.unique(fundus_patch_array, return_counts=True))

    # sys.exit()

    print(fundus_patch_array.shape)
    # for idx in range(fundus_patch_array.shape[0]):
        
    #     img_patch = fundus_patch_array[idx, 0, ...]
    #     plt.figure()
    #     plt.imshow(img_patch)
    
    # plt.show()



if __name__ == '__main__':
    main()