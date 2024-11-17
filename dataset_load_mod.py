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
from skimage.color import rgb2gray, rgba2rgb
from skimage.filters import threshold_otsu
from math import ceil, floor
from skimage.io import imread
from tensorflow.python.ops.gen_batch_ops import batch

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
    
    patch_num = steps_x * steps_y

    patch_x_arr = np.zeros((patch_num, patch_dim_x, patch_dim_y, img_input.shape[-1]), dtype=np.float32)
    patch_y_arr = None

    if img_label is not None:
        patch_y_arr = np.zeros((patch_num, patch_dim_x, patch_dim_y, img_label.shape[-1]), dtype=np.float32)


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
                img_y_patch = img_label[x_init:x_end, img_res_y - patch_dim_y : img_res_y]

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


def get_unique_filenames_in_folder(folder):
    """
    Iterates through a given folder and its subfolders,
    and retrieves the unique names of all files (not folders),
    excluding hidden/system files.
    
    Args:
        folder (str): Path to the folder.
    
    Returns:
        set: A set containing the unique file names in the folder.
    """
    unique_filenames = set()
    
    # Walk through the directory structure
    for root, _, files in os.walk(folder):
        for file in files:
            # Exclude hidden/system files (e.g., .DS_Store)
            if not file.startswith('.'):
                unique_filenames.add(file)
    
    return unique_filenames


def preprocess_sample(sample_fundus, sample_octa):
    """
    Preprocess fundus and OCTA samples:
    - Normalize fundus to the range [-1, 1].
    - Convert OCTA to grayscale and normalize to the range [-1, 1].
    
    Args:
        sample_fundus (numpy.ndarray): Fundus image array.
        sample_octa (numpy.ndarray): OCTA image array.
    
    Returns:
        tuple: Preprocessed (sample_fundus, sample_octa).
    """
    # Normalize fundus image to range [-1, 1]
    sample_fundus = (sample_fundus / 255.0) * 2.0 - 1.0

    # # Check if OCTA is RGBA, and convert to RGB if needed
    # if sample_octa.shape[-1] == 4:  # If 4 channels (RGBA)
    #     sample_octa = rgba2rgb(sample_octa)

    # # Convert OCTA to grayscale (if it has multiple channels)
    # if len(sample_octa.shape) == 3 and sample_octa.shape[-1] > 1:
    #     sample_octa = rgb2gray(sample_octa)

    # Normalize grayscale OCTA to range [-1, 1]
    sample_octa = (sample_octa / 255.0) * 2.0 - 1.0

    return sample_fundus, sample_octa



# @tf.function
def load_data_cases(data_fold = "", list_cases=[], eye_list= ["OS", "OD"], regions_list=['macula', 'disc'], patch_dim= (128, 128), random_state_patches = 0
, num_patches = 15, fundus_channels = 3, oct_channels = 4):


    fundus_data_fold = "{0}/{1}".format(data_fold, 'fundus')
    octa_data_fold = "{0}/{1}".format(data_fold, 'octa')

    fundus_list = get_unique_filenames_in_folder(fundus_data_fold)

    sample_count = 0

    array_fundus, array_octa = None, None


    print(fundus_list)

    for img_id in fundus_list:

        img_number = int(img_id.split("_")[0])

        # This checks that the id number is within 
        # the training, validation or test set it belongs to
        if img_number not in list_cases:
            continue

        for eye_region in regions_list:

            try:

                eye_path_fundus = "{0}/{1}/{2}".format(fundus_data_fold, eye_region, img_id)
                eye_path_octa = "{0}/{1}/{2}".format(octa_data_fold, eye_region, img_id)

                sample_fundus = io.imread(eye_path_fundus)
                sample_octa = io.imread(eye_path_octa)

                sample_fundus, sample_octa = preprocess_sample(sample_fundus, sample_octa)


                if sample_fundus.shape[0] < patch_dim[0] or sample_fundus.shape[1] < patch_dim[1]:
                    continue

                fundus_disc_patch, octa_disc_patch = break_img_to_patches(img_input=sample_fundus, img_label=sample_octa
                , patch_size=patch_dim, return_dataset=False, stride=1)

                print(fundus_disc_patch.shape)
                print(octa_disc_patch.shape)

                if sample_count == 0:
                    sample_count = 1
                    
                    array_fundus = fundus_disc_patch
                    array_octa = octa_disc_patch


                else:
                    
                    array_fundus = np.concatenate((array_fundus, fundus_disc_patch))
                    array_octa = np.concatenate((array_octa, octa_disc_patch))


            except Exception as e:
                continue
            

    return array_fundus, array_octa



def main():
    fundus_sample_x = imread("data/nasa_05_23_21/sub-0000053/ses-01/fundus/000053__66_Image_OS.jpg")

    img_patches = break_img_to_patches(img_input=fundus_sample_x, img_label=None
    , patch_size=(512, 512), return_dataset=False, stride= 2)


    fundus_patch_array = img_patches[0].astype(int)

    img_out = put_img_back_from_patches(fundus_patch_array, fundus_sample_x.shape, stride=4)

    print(img_out.shape)
    plt.imshow(img_out.astype(int))
    plt.show()


if __name__ == '__main__':
    main()