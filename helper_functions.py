from __future__ import division
import os
import time
from glob import glob
from utils import *
from sklearn import preprocessing

def read_images(c_dim,config):
    is_grayscale = (c_dim == 1)
    real_data = glob(os.path.join("./data/denoise/grayOriginal", "*.jpg"))
    noise_data = glob(os.path.join("./data/denoise/graygauss", "*.jpg"))

    real = [get_image(img_file, config.image_size, is_crop=False, is_grayscale=is_grayscale) for img_file in real_data]
    noise = [get_image(img_file, config.image_size, is_crop=False, is_grayscale=is_grayscale) for img_file in noise_data]

    if is_grayscale:
        reals = np.array(real).astype(np.float32)[:,:,:,None]
        noises = np.array(noise).astype(np.float32)[:,:,:,None]
    else:
        reals = np.array(real).astype(np.float32)
        noises = np.array(noise).astype(np.float32)

    return reals, noises

def read_images2(c_dim,config):
    is_grayscale = (c_dim == 1)
    # real_data = glob(os.path.join("./data/simsun", "*.npy"))
    # noise_data = glob(os.path.join("./data/jg", "*.npy"))
    #
    # real = [get_image(img_file, config.image_size, is_crop=False, is_grayscale=is_grayscale) for img_file in real_data]
    # noise = [get_image(img_file, config.image_size, is_crop=False, is_grayscale=is_grayscale) for img_file in noise_data]

    if is_grayscale:
        noises = np.load('./data/simsun80.npy').astype(np.float32)[:,:,:,None]
        reals = np.load('./data/jg.npy').astype(np.float32)[:,:,:,None]

        # noises = np.load('./data/simsun80_norm.npy').astype(np.float32)[:, :, :, None]
        # reals = np.load('./data/jgnorm.npy').astype(np.float32)[:, :, :, None]
        #reals = np.array(real).astype(np.float32)[:,:,:,None]
        #noises = np.array(noise).astype(np.float32)[:,:,:,None]
    else:
        reals = np.array(real).astype(np.float32)
        noises = np.array(noise).astype(np.float32)

    return reals, noises