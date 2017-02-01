# -*- coding: utf-8 -*-
"""
Created on Wed Dec 28 17:52:25 2016

@author: lenovo
"""

import numpy as np  
from skimage import io, color
from scipy import misc
import matplotlib.image as mpimg

import tensorlayer
print ("import successfully")

#==============================================================================
# orginal = np.load('simsun.npy')
# jp = np.load('jg.npy')
# sim80 = np.load('simsun80.npy')
# re = np.load('targetbitmap.npy')
# simsun80 = np.zeros([np.size(orginal,0),80,80])
# re80 = np.zeros([np.size(re,0),80,80])
#==============================================================================
# for i in range(np.size(orginal,0)):
#     #io.imsave('Original/'+np.str(i)+'.jpg',orginal[i])
#     img = mpimg.imread('Original/'+np.str(i)+'.jpg')
#     new_sz = misc.imresize(img, 0.5)
#     #io.imsave('Original80/'+np.str(i)+'.jpg',new_sz)
#     simsun80[i] = new_sz
#
# simsun80 = simsun80.astype(np.uint8)
# np.save('simsun80',simsun80)

#==============================================================================
# re = np.load('result255.npy')    
# re = re.astype(np.float32)
# for i in range(np.size(re,0)):
#     re[i] = re[i]/255
#     io.imsave('result/'+np.str(i)+'.jpg',re[i])
# np.save('resultnorm',re)
#==============================================================================

# for i in range(np.size(jp,0)):
#     jp[i] = jp[i]//255
#     io.imsave('jgnorm/'+np.str(i)+'.jpg',jp[i]*255)
#
# jpnorm = jp.astype(np.uint8)
# np.save('jgnorm',jpnorm)


orginal = np.load('gbxs.npy')
io.imsave('gbxs.jpg',orginal[0])