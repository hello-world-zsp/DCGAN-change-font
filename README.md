This code works for Chinese fonts changing.
#
- The original font images should be .jpg files, the same for the objective font images. The are suggested to be saved in ./data.
- If your data is saved as a whole '.npy', format-converting functions (eg. npy2jpg.py) are also provided in the folder 'data'.

#
- helper_functions.py include reading image functions.

- utils.py is copy from [https://github.com/carpedm20/DCGAN-tensorflow](https://github.com/carpedm20/DCGAN-tensorflow), which is used to read png files of given directory.
- main.py is used to train the whole network. 

`python main.py  --batch_size 64 --image_size 80 --epoch 100 --learning_rate 0.00001 --c_dim 1`

--dataset, the folder name which you save your images, ./data/stl_binary/real_images.
--batch_size, this number is related to final sample images, if you set batch_size is 100, you need change the dcgan.py file line 262 [8, 8] to [10, 10].
--c_dim, which means gray(1) or rgb(3).
--image_size
--epoch
--learning_rate

