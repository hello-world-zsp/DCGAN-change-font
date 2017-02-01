# -*-coding:utf-8-*-
## 改过虚拟环境的配置，在project interpreter里面 现在用的是Python3的虚拟环境
import tensorflow as tf
import os
import time
from ops import *
from helper_functions import *

class MYDCGAN (object):
    def __init__(self, sess, img_size = 80, is_crop = True,
                 batch_size=100, sample_size = 64, output_size=80,
                 y_dim=None, z_dim = 100, gf_dim=64, df_dim=16,
                 gfc_dim=1024, dfc_dim=1024, c_dim = 1, dataset_name='default',
                 checkpoint_dir=None, sample_dir=None,
                 num_filter = 16, leak = 0.1, stride = [1, 1, 1, 1]):
        """

        Args:
            sess: TensorFlow session
            batch_size: The size of batch. Should be specified before training.
            output_size: (optional) The resolution in pixels of the images. [64]
            y_dim: (optional) Dimension of dim for y. [None]
            z_dim: (optional) Dimension of dim for Z. [100]
            gf_dim: (optional) Dimension of gen filters in first conv layer. [64]
            df_dim: (optional) Dimension of discrim filters in first conv layer. [64]
            gfc_dim: (optional) Dimension of gen units for for fully connected layer. [1024]
            dfc_dim: (optional) Dimension of discrim units for fully connected layer. [1024]
            c_dim: (optional) Dimension of image color. For grayscale input, set to 1. [3]
        """
        self.sess = sess
        self.is_crop = is_crop
        self.is_grayscale = (c_dim == 1)
        self.batch_size = batch_size
        self.img_size = img_size
        self.sample_size = sample_size
        self.output_size = output_size

        self.y_dim = y_dim
        self.z_dim = z_dim          #initial noise for g_model

        self.gf_dim = gf_dim
        self.df_dim = df_dim

        self.gfc_dim = gfc_dim
        self.dfc_dim = dfc_dim

        self.c_dim = c_dim
        self.num_filter = num_filter
        self.leak = leak
        self.stride = stride
        self.alpha = 0
        self.alpha_step = 0.05


        # batch normalization : deals with poor initialization helps gradient flow
        self.d_bn1 = batch_norm(name='d_bn1')
        self.d_bn2 = batch_norm(name='d_bn2')
        self.d_bn3 = batch_norm(name='d_bn3')

        self.g_bn0 = batch_norm(name='g_bn0')
        self.g_bn1 = batch_norm(name='g_bn1')
        self.g_bn2 = batch_norm(name='g_bn2')
        self.g_bn3 = batch_norm(name='g_bn3')
        self.g_bn4 = batch_norm(name='g_bn4')

        self.dataset_name = dataset_name
        self.checkpoint_dir = checkpoint_dir
        self.build_model()

    def generator(self, gdata):
        with tf.variable_scope("generator") as scope:
            if self.stride[1]>1:
                s2 = self.img_size // self.stride[1]  # 改，s2需要Int型
                s4 = self.img_size // self.stride[1]**2
            else:
                s2 = self.img_size
                s4 = self.img_size

            stddev = 0.02
            #gdata = self.g_bn0(gdata)
            with tf.variable_scope('g_conv1') as scope:
                w = tf.get_variable('w', [4, 4, self.c_dim, self.num_filter],
                                    initializer=tf.random_normal_initializer(stddev=stddev))
                gconv = tf.nn.conv2d(gdata, w, strides=self.stride,
                                     padding='SAME')
                biases = tf.get_variable('biases', [self.num_filter],
                                         initializer=tf.constant_initializer(0.0))
                bias = tf.nn.bias_add(gconv, biases)
                #bias = self.g_bn1(bias)
                gconv1 = tf.nn.relu(bias, name=scope.name)

            with tf.variable_scope('g_conv2') as scope:
                w = tf.get_variable('w', [4, 4, self.num_filter, self.num_filter * 2],
                                    initializer=tf.random_normal_initializer(stddev=stddev))
                gconv = tf.nn.conv2d(gconv1, w, strides=self.stride,
                                     padding='SAME')
                biases = tf.get_variable('biases', [self.num_filter * 2],
                                         initializer=tf.constant_initializer(0.0))
                bias = tf.nn.bias_add(gconv, biases)
                #bias = self.g_bn2(bias)
                gconv2 = tf.nn.relu(bias, name=scope.name)

            # with tf.variable_scope('g_conv3') as scope:
            #     w = tf.get_variable('w', [4, 4, self.num_filter * 2, self.num_filter * 4],
            #                         initializer=tf.random_normal_initializer(stddev=stddev))
            #     gconv = tf.nn.conv2d(gconv2, w, strides=self.stride,
            #                          padding='SAME')
            #     biases = tf.get_variable('biases', [self.num_filter * 4],
            #                              initializer=tf.constant_initializer(0.0))
            #     bias = tf.nn.bias_add(gconv, biases)
            #     gconv3 = tf.nn.relu(bias, name=scope.name)
            #
            # with tf.variable_scope('g_conv4') as scope:
            #     w = tf.get_variable('w', [4, 4, self.num_filter * 4, 1],
            #                         initializer=tf.random_normal_initializer(stddev=stddev))
            #     gconv = tf.nn.conv2d(gconv3, w, strides=self.stride,
            #                          padding='SAME')
            #     biases = tf.get_variable('biases', [1],
            #                              initializer=tf.constant_initializer(0.0))
            #     bias = tf.nn.bias_add(gconv, biases)
            #     gconv4 = tf.nn.relu(bias, name=scope.name)

            # with tf.variable_scope('g_deconv0') as scope:
            #     w = tf.get_variable('w', [4, 4, self.num_filter * 2, self.num_filter * 4],
            #                         initializer=tf.random_normal_initializer(stddev=stddev))
            #     deconv = tf.nn.conv2d_transpose(gconv3, w,
            #                                     output_shape=[self.batch_size, s4, s4, self.num_filter*2],
            #                                     strides=self.stride)
            #
            #     biases = tf.get_variable('biases', [self.num_filter*2],
            #                              initializer=tf.constant_initializer(0.0))
            #     deconv0 = tf.nn.bias_add(deconv, biases)
            #
            with tf.variable_scope('g_deconv1') as scope:
                w = tf.get_variable('w', [4, 4, self.num_filter, self.num_filter * 2],
                                    initializer=tf.random_normal_initializer(stddev=stddev))
                deconv = tf.nn.conv2d_transpose(gconv2, w,
                                                output_shape=[self.batch_size, s2, s2, self.num_filter],
                                                strides=self.stride)

                biases = tf.get_variable('biases', [self.num_filter],
                                         initializer=tf.constant_initializer(0.0))
                deconv1 = tf.nn.bias_add(deconv, biases)
                #deconv1 = self.g_bn3(deconv1)
                deconv1 = tf.nn.relu(deconv1)

            with tf.variable_scope('g_deconv2') as scope:
                w = tf.get_variable('w', [4, 4, self.c_dim, self.num_filter],
                                    initializer=tf.random_normal_initializer(stddev=stddev))
                deconv = tf.nn.conv2d_transpose(deconv1, w,
                                                output_shape=[self.batch_size, self.img_size, self.img_size, self.c_dim],
                                                strides=self.stride)
                biases = tf.get_variable('biases', [self.c_dim],
                                         initializer=tf.constant_initializer(0.0))
                deconv2 = tf.nn.bias_add(deconv, biases)
                #deconv2 = self.g_bn4(deconv2)

            temp1 = tf.nn.tanh(deconv2)

            return temp1


    def discriminator(self,ddata, reuse=False):
        with tf.variable_scope("discriminator") as scope:  # 自己加的with，解决196行adam问题
            if reuse:
                # tf.get_variable_scope().reuse_variables()
                scope.reuse_variables()  # sharing variables

            stddev = 0.002

            # h0 = lrelu(conv2d(ddata, self.df_dim, name='d_h0_conv'))
            # h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim * 2, name='d_h1_conv')))
            # h2 = lrelu(self.d_bn2(conv2d(h1, self.df_dim * 4, name='d_h2_conv')))
            # h3 = lrelu(self.d_bn3(conv2d(h2, self.df_dim * 8, name='d_h3_conv')))
            # h4 = linear(tf.reshape(h3, [self.batch_size, -1]), 1, 'd_h3_lin')
            #
            # return tf.nn.sigmoid(h4), h4

            with tf.variable_scope('d_conv1') as scope:
                w = tf.get_variable('w', [4, 4, self.c_dim, self.num_filter],
                                    initializer=tf.truncated_normal_initializer(stddev=stddev))
                dconv = tf.nn.conv2d(ddata, w, strides=self.stride,
                                     padding='SAME')
                biases = tf.get_variable('biases', [self.num_filter],
                                         initializer=tf.constant_initializer(0.0))
                bias = tf.nn.bias_add(dconv, biases)
                dconv1 = tf.maximum(bias, self.leak * bias)


            with tf.variable_scope('d_conv2') as scope:
                w = tf.get_variable('w', [4, 4, self.num_filter, self.num_filter * 2],
                                    initializer=tf.truncated_normal_initializer(stddev=stddev))
                dconv = tf.nn.conv2d(dconv1, w, strides=self.stride,
                                     padding='SAME')
                biases = tf.get_variable('biases', [self.num_filter * 2],
                                         initializer=tf.constant_initializer(0.0))
                bias = tf.nn.bias_add(dconv, biases)
                dconv2 = tf.maximum(bias, self.leak * bias)

            with tf.variable_scope('d_conv3') as scope:
                w = tf.get_variable('w', [4, 4, self.num_filter * 2, self.num_filter * 4],
                                    initializer=tf.truncated_normal_initializer(stddev=stddev))
                dconv = tf.nn.conv2d(dconv2, w, strides=self.stride,
                                     padding='SAME')
                biases = tf.get_variable('biases', [self.num_filter * 4],
                                         initializer=tf.constant_initializer(0.0))
                bias = tf.nn.bias_add(dconv, biases)
                dconv3 = tf.maximum(bias, self.leak * bias)

            with tf.variable_scope('d_conv4') as scope:
                w = tf.get_variable('w', [4, 4, self.num_filter * 4, self.num_filter * 8],
                                    initializer=tf.truncated_normal_initializer(stddev=stddev))
                dconv = tf.nn.conv2d(dconv3, w, strides=self.stride,
                                     padding='SAME')
                biases = tf.get_variable('biases', [self.num_filter * 8],
                                         initializer=tf.constant_initializer(0.0))
                bias = tf.nn.bias_add(dconv, biases)
                dconv4 = tf.maximum(bias, self.leak * bias)

            with tf.variable_scope('d_local1') as scope:
                local_in = tf.reshape(dconv4, [self.batch_size, -1])
                shape = local_in.get_shape().as_list()

                w = tf.get_variable('w', [shape[1], 1], tf.float32,
                                    tf.random_normal_initializer(stddev=stddev))
                biases = tf.get_variable("biases", [1],
                                         initializer=tf.constant_initializer(0.0))
                dlocal = tf.matmul(local_in, w) + biases

        return tf.nn.sigmoid(dlocal), dlocal

    def build_model(self):
        self.noise_images = tf.placeholder(tf.float32, [self.batch_size]
                                      + [self.img_size, self.img_size, self.c_dim], name='noise_images')
        self.real_images = tf.placeholder(tf.float32, [self.batch_size]
                                     + [self.img_size, self.img_size, self.c_dim], name='real_images')

        self.G = self.generator(self.noise_images)
        self.D, self.D_logots = self.discriminator(self.real_images)
        self.D_, self.D_logots_ = self.discriminator(self.G, reuse=True)

        self.G_sum = image_summary("G", self.G)

        self.d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.D_logots, tf.ones_like(self.D)))
        self.d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.D_logots_, tf.zeros_like(self.D_)))
        g_loss1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.D_logots_, tf.ones_like(self.D_)))
        #tempG = tf.reshape(self.G,[self.batch_size,self.img_size*self.img_size])
        #tempReal = tf.reshape(self.real_images,[self.batch_size,self.img_size*self.img_size])
        g_loss2 = tf.reduce_mean(tf.abs((self.G-self.real_images)))     #大于0
        self.g_loss = self.alpha * g_loss1 + g_loss2

        self.d_loss = self.d_loss_real + self.d_loss_fake

        self.d_loss_real_sum = scalar_summary("d_loss_real", self.d_loss_real)
        self.d_loss_fake_sum = scalar_summary("d_loss_fake", self.d_loss_fake)
        self.g_loss_sum = scalar_summary("g_loss", self.g_loss)
        self.d_loss_sum = scalar_summary("d_loss", self.d_loss)

        t_vars = tf.trainable_variables()

        self.d_vars = [var for var in t_vars if 'd_' in var.name]
        self.g_vars = [var for var in t_vars if 'g_' in var.name]

        self.saver = tf.train.Saver()


    # def train(sess, G, d_loss, d_vars, g_loss, g_vars, saver, c_dim=1):
    def train(self, config):
        # noise_images = tf.placeholder(tf.float32, [self.batch_size]
        #                               + [self.img_size, self.img_size, self.c_dim], name='noise_images')
        # real_images = tf.placeholder(tf.float32, [self.batch_size]
        #                              + [self.img_size, self.img_size, self.c_dim], name='real_images')

        # G = self.generator(self.noise_images, self.img_size, self.batch_size, self.c_dim, self.num_filter)
        # D, D_logots = self.discriminator(self.real_images, self.batch_size, self.c_dim, self.num_filter, leak)
        # D_, D_logots_ = self.discriminator(G, self.batch_size, self.c_dim, self.num_filter, leak, reuse=True)
        #
        # d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(D_logots, tf.ones_like(D)))
        # d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(D_logots_, tf.zeros_like(D_)))
        # g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(D_logots_, tf.ones_like(D_)))
        #
        # d_loss = d_loss_real + d_loss_fake
        #
        # t_vars = tf.trainable_variables()
        #
        # d_vars = [var for var in t_vars if 'd_' in var.name]
        # g_vars = [var for var in t_vars if 'g_' in var.name]
        #
        # saver = tf.train.Saver()

        reals, noises = read_images2(self.c_dim,config)

        d_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1).minimize(self.d_loss, var_list=self.d_vars)
        #g_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1).minimize(self.g_loss, var_list=self.g_vars)
        g_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1).minimize(self.g_loss,var_list=self.g_vars)

        tf.initialize_all_variables().run()

        # self.g_sum = merge_summary([self.d__sum,
        #                        self.G_sum, self.d_loss_fake_sum, self.g_loss_sum])
        # self.d_sum = merge_summary([self.d_sum, self.d_loss_real_sum, self.d_loss_sum])
        self.g_sum = merge_summary([self.G_sum, self.d_loss_fake_sum, self.g_loss_sum])
        self.d_sum = merge_summary([self.d_loss_real_sum, self.d_loss_sum])
        self.writer = SummaryWriter("./logs", self.sess.graph)

        start_time = time.time()
        counter = 0

        sample_images = reals[0:self.batch_size]
        sample_z = noises[0:self.batch_size]

        model_name = "MYDCGAN.model"
        model_dir = "%s_%s_%s" % (config.dataset, config.batch_size, config.image_size)
        checkpoint_dir = os.path.join('./checkpoint', model_dir)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        for epoch in range(config.epoch):

            num_batch = len(reals) // config.batch_size
            print('num_batch', num_batch)

            for idx in range(0, num_batch):

                batch_images = reals[idx * config.batch_size:(idx + 1) * config.batch_size]
                batch_z = noises[idx * config.batch_size:(idx + 1) * config.batch_size]

                if counter > 1000:
                    # update D
                    _, summary_str = self.sess.run([d_optim, self.d_sum], feed_dict={self.real_images: batch_images, self.noise_images: batch_z})
                    self.writer.add_summary(summary_str, counter)

                # update G
                _, summary_str = self.sess.run([g_optim,self.g_sum],
                                               feed_dict={self.real_images: batch_images, self.noise_images: batch_z})
                # update G twice
                _, summary_str = self.sess.run([g_optim, self.g_sum],
                                               feed_dict={self.real_images: batch_images, self.noise_images: batch_z})
                self.writer.add_summary(summary_str, counter)
                self.alpha = (counter -1000) * self.alpha_step

                # # update G again
                # _, summary_str = self.sess.run([g_optim,self.g_sum], feed_dict={self.noise_images: batch_z})
                # self.writer.add_summary(summary_str, counter)

                errD_fake = self.d_loss_fake.eval({self.noise_images: batch_z})
                errD_real = self.d_loss_real.eval({self.real_images: batch_images})
                #errG = self.g_loss.eval({self.noise_images: batch_z})
                errG = self.g_loss.eval({self.real_images: batch_images, self.noise_images: batch_z})

                counter += 1
                print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" % (epoch,
                                                                                          idx, num_batch,
                                                                                          time.time() - start_time,
                                                                                          errD_fake + errD_real, errG))

                if counter == 1:
                    save_images(sample_z, [10, 10], 'results/' + 'noise' + str(counter) + '.jpg')
                    save_images(sample_images, [10, 10], 'results/' + 'original' + str(counter) + '.jpg')
                if np.mod(counter, 100) == 1:
                    samples, loss1, loss2 = self.sess.run([self.G, self.d_loss,
                                                      self.g_loss], feed_dict={self.noise_images: sample_z,
                                                                               self.real_images: sample_images})

                    save_images(samples, [10, 10], 'results/' + 'denoise' + str(counter) + '.jpg')
                    print("[Sample] d_loss: %.8f, g_loss: %.8f" % (loss1, loss2))

                if np.mod(counter, 500) == 2:
                    self.saver.save(self.sess, os.path.join(checkpoint_dir, model_name), global_step=counter)


