import os
import numpy as np

from model_my import MYDCGAN
from utils import pp, visualize, to_json

import tensorflow as tf

flags = tf.app.flags
flags.DEFINE_integer("epoch", 100, "Epoch to train [100]")
flags.DEFINE_float("learning_rate", 0.0002, "Learning rate of for adam [0.0002]")
flags.DEFINE_float("learning_rate_g", 0.002, "Learning rate of for adam [0.002]")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
flags.DEFINE_integer("train_size", np.inf, "The size of train images [np.inf]")
flags.DEFINE_integer("batch_size", 64, "The size of batch images [64]")
flags.DEFINE_integer("image_size", 80, "The size of image to use (will be center cropped) [80]")
flags.DEFINE_integer("output_size", 80, "The size of the output images to produce [80]")
flags.DEFINE_integer("c_dim", 1, "Dimension of image color. [1]")
flags.DEFINE_string("dataset", "celebA", "The name of dataset [celebA, mnist, lsun]")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("sample_dir", "samples", "Directory name to save the image samples [samples]")
flags.DEFINE_boolean("is_train", False, "True for training, False for testing [False]")
flags.DEFINE_boolean("is_crop", False, "True for training, False for testing [False]")
flags.DEFINE_boolean("visualize", False, "True for visualizing, False for nothing [False]")
FLAGS = flags.FLAGS

def main(_):
    pp.pprint(flags.FLAGS.__flags)

    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)
    if not os.path.exists(FLAGS.sample_dir):
        os.makedirs(FLAGS.sample_dir)

    with tf.Session() as sess:
        dcgan = MYDCGAN(sess,
                      img_size=FLAGS.image_size,
                      batch_size=FLAGS.batch_size,
                      output_size=80,           #output image size 28*28
                      c_dim=1,                  #color dim:1(gray channel)
                      dataset_name=FLAGS.dataset,
                      is_crop=FLAGS.is_crop,    #crop boarder
                      checkpoint_dir=FLAGS.checkpoint_dir,  #log dir
                      sample_dir=FLAGS.sample_dir)

        # if FLAGS.is_train:
        #     dcgan.train(FLAGS)
        # else:
        #     dcgan.load(FLAGS.checkpoint_dir)

        dcgan.train(FLAGS)

        OPTION = 1
        #visualize(sess, dcgan, FLAGS, OPTION)

if __name__ == '__main__':
    tf.app.run()
