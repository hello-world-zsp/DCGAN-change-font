#-*-coding:utf-8-*-
from skimage import io, color
import numpy as np

def convert_gray(f):
    rgb = io.imread(f)
    return color.rgb2gray(rgb)

str = 'denoise/resizephoto/*.JPEG'
coll = io.ImageCollection(str, load_func=convert_gray)
for i in range(len(coll)):
    io.imsave('denoise/grayOriginal/'+np.str(i)+'.jpg',coll[i])  #循环保存图片