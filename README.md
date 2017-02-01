This code works for Chinese fonts changing.
#
- The original font images should be .jpg files or '.npy' as a whole, the same for the objective font images. The are suggested to be saved in ./data.
- format-converting functions (eg. npy2jpg.py) are also provided in the folder 'data'.

#
- helper_functions.py include reading image functions.

- utils.py is copy from [https://github.com/carpedm20/DCGAN-tensorflow](https://github.com/carpedm20/DCGAN-tensorflow), which is used to read png files of given directory.
- main.py is used to train the whole network. 
- model_my.py is the file which you can adjust the DCGAN network in detail.
#
`python main.py --batch_size 20 --image_size 80 --epoch 20 --learning_rate 0.000001 --c_dim 1`

- --batch_size, if you are in GPU mode, too large batch_size can cause your GPU out of memory.
- --c_dim, which means gray(1) or rgb(3).
- --image_size, 80 in my work.
- --epoch
- --learning_rate

- --dataset, if your original data are images instead of npy, add this parameter. It is the folder name which you save your images, eg. ./data/dataset/original_images.
