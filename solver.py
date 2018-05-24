import tensorflow as tf
import util
import os
import numpy as np
import model
from torchvision.utils import save_image
import torch

class Solver(object):
    """docstring for Solver"""
    def __init__(self, data, config):
        super(Solver, self).__init__()

        self.data = data # data is a generator

        tf.set_random_seed(config.seed)
        np.random.seed(config.seed)
        os.environ['CUDA_VISIBLE_DEVICES'] = config.gpus

        self.crop_size = config.crop_size
        self.image_size = config.image_size
        self.c_dim = config.c_dim
        self.g_conv_num = config.g_conv_num
        self.d_conv_num = config.d_conv_num
        self.g_res_num = config.g_res_num
        self.lambda_cls = config.lambda_cls
        self.lambda_rec = config.lambda_rec
        self.lambda_gp = config.lambda_gp
        self.lr = config.lr
        self.resume = config.resume
        self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth = True

        self.epochs = config.epochs
        self.n_critic = config.n_critic
        self.length = len(data)
        self.result_dir = config.result_dir


        if tf.gfile.Exists(config.log_dir):
            tf.gfile.DeleteRecursively(config.log_dir)
        else:
            tf.gfile.MakeDirs(config.log_dir)

        self.model_save_dir = config.model_save_dir
        self.model_name = config.model_name
        self.selected_attrs = config.selected_attrs

        self.build_model()

        self.summary_writer = tf.summary.FileWriter(logdir=config.log_dir, graph=tf.get_default_graph())
        self.saver = tf.train.Saver(max_to_keep=5)

    def preprocess(self, image):

        image_flip = tf.image.random_flip_left_right(image) # np.flip(A, axis=1)
        image_crop = tf.random_crop(image_flip, size=(self.crop_size, self.crop_size, 3)) # random crop (178, 178)
        image_resize = tf.image.resize_images(image_crop, size=(self.image_size, self.image_size)) # resize (128, 128)
        image_normalize = image_resize / 255. # normalize to (0, 1)

        return image_normalize

    def gan_loss(self, logits, labels):
        return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits,labels=tf.to_float(labels)))

    def recon_loss(self, image1, image2):
        return tf.reduce_mean(tf.abs(image1 - image2))

    def gradient_penalty(self, y, x):
        """
        compute gradient penalty: (norm(dy/dx) - 1)^2
        """
        gradients = tf.gradients(ys=y, xs=x)[0]
        slopes = tf.sqrt(tf.reduce_sum(gradients**2, axis=[1,2,3]))
        gradient_penalty = tf.reduce_mean((slopes - 1.)**2)

        return gradient_penalty

    def build_model(self):

        self.images_real = tf.placeholder(tf.float32, shape=(None, None, None, 3), name='images')
        self.labels_src = tf.placeholder(tf.int64, shape=(None, self.c_dim), name='labels_src')
        self.labels_trg = tf.placeholder(tf.int64, shape=(None, self.c_dim), name='labels_trg')
        #############################################################################################################
        ## image size (218, 178), random flip left and right, random crop to (178, 178), then resize to (128, 128) ##
        #############################################################################################################
        self.images_real_normalized = tf.map_fn(lambda x: self.preprocess(x), elems=self.images_real, back_prop=False, parallel_iterations=10) # (None, image_size, image_size, 3)

        self.logits_real, self.logits_class_real = model.discriminator(self.images_real_normalized, self.d_conv_num, self.c_dim, reuse=False) # (None, H/64, W/64, 1), (None, c_dim)

        self.loss_real_d, self.loss_real_cls = -tf.reduce_mean(self.logits_real), self.gan_loss(self.logits_class_real, self.labels_src)

        self.images_fake = model.generator(self.images_real_normalized, self.labels_trg, self.g_conv_num, self.g_res_num, reuse=False)

        self.images_reconst = model.generator(self.images_fake, self.labels_src, self.g_conv_num, self.g_res_num, reuse=True)

        self.loss_reconst = self.recon_loss(self.images_fake, self.images_reconst)

        self.logits_fake, self.logits_class_fake = model.discriminator(self.images_fake, self.d_conv_num, self.c_dim, reuse=True) # (None, H/64, W/64, 1), (None, c_dim)

        self.loss_fake_d, self.loss_fake_cls = tf.reduce_mean(self.logits_fake), self.gan_loss(self.logits_class_fake, self.labels_trg)

        self.alpha = tf.random_normal(shape=tf.shape(self.labels_trg[:, 0, None, None, None])) # (None, 1, 1, 1)
        self.x_hat = self.alpha * self.images_real_normalized + (1-self.alpha) * self.images_fake
        self.logits_hat, _ = model.discriminator(self.x_hat, self.d_conv_num, self.c_dim, reuse=True)
        self.grad_penalty = self.gradient_penalty(self.logits_hat, self.x_hat)
        

        self.loss_d = self.loss_real_d + self.loss_fake_d + self.lambda_cls * self.loss_real_cls + self.lambda_gp * self.grad_penalty
        self.loss_g = -self.loss_fake_d + self.lambda_rec * self.loss_reconst + self.lambda_cls * self.loss_fake_cls

        global_step = tf.Variable(0, trainable=False)

        lr = tf.train.exponential_decay(self.lr, global_step, 1e4, 0.1, staircase=False)

        self.opt_d = tf.train.AdamOptimizer(lr, beta1=0.5, beta2=0.999)
        self.opt_g = tf.train.AdamOptimizer(lr, beta1=0.5, beta2=0.999)

        self.params_d = tf.trainable_variables(scope='discriminator')
        self.params_g = tf.trainable_variables(scope='generator')

        self.train_op_d = self.opt_d.minimize(self.loss_d, var_list=self.params_d)
        self.train_op_g = self.opt_g.minimize(self.loss_g, var_list=self.params_g)

        # summary op
        self.loss_real_d_summary, self.loss_real_cls_summary = tf.summary.scalar('D_score_for_real', self.loss_real_d), tf.summary.scalar('class_score_for_real', self.loss_real_cls)
        self.loss_fake_d_summary, self.loss_fake_cls_summary = tf.summary.scalar('D_score_for_fake', self.loss_fake_d), tf.summary.scalar('class_score_for_fake', self.loss_fake_cls)
        self.grad_penalty_summary = tf.summary.scalar('grad_penalty', self.grad_penalty)
        self.loss_reconst_summary = tf.summary.scalar('reconstruct_loss', self.loss_reconst)

        self.images_real_summary = tf.summary.image('real_images', self.images_real)
        self.images_fake_summary = tf.summary.image('fake_images', self.images_fake)
        self.images_reconst_summary = tf.summary.image('reconstruct_images', self.images_reconst)

        self.summary_loss = tf.summary.merge([self.loss_real_d_summary, self.loss_real_cls_summary,
                                        self.loss_fake_d_summary, self.loss_fake_cls_summary, 
                                        self.grad_penalty_summary, self.loss_reconst_summary])
        self.summary_images = tf.summary.merge([self.images_real_summary, self.images_fake_summary, self.images_reconst_summary])

    def checkpoint_save(self, sess, step):
        self.saver.save(sess, os.path.join(self.model_save_dir, self.model_name), step)


    def checkpoint_load(self, sess):
        
        ckpt = tf.train.get_checkpoint_state(self.model_save_dir)        
        if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(sess, ckpt.model_checkpoint_path)
            return True
        else:
            return False

    def train(self):

        with tf.Session(config=self.config) as sess:

            sess.run(tf.global_variables_initializer())

            print "Start training......"

            if self.resume and self.checkpoint_load(sess):
                print "Resume from last checkpoint......"
            else:
                print "No model saved, training from beginning......"


            for epoch in range(self.epochs):

                for step, (images, labels) in enumerate(self.data):

                    labels_target = labels.copy()
                    np.random.shuffle(labels_target)

                    feed_dict = {self.images_real: images, self.labels_src: labels, self.labels_trg: labels_target}

                    for _ in range(self.n_critic):
                        _, loss_real_d, loss_fake_d, loss_real_cls, grad_penalty = sess.run([self.train_op_d, self.loss_real_d, self.loss_fake_d, 
                                                                                             self.loss_real_cls, self.grad_penalty], feed_dict)

                    _, loss_fake_dg, loss_fake_cls, loss_reconst = sess.run([self.train_op_g, self.loss_fake_d, self.loss_fake_cls, self.loss_reconst], feed_dict)


                    print "[Discriminator] epoch: [{:02d}/{}] step: [{:05d}/{}], loss real: [{:.4f}], loss fake:  [{:.4f}],  class loss real: [{:.4f}]".format(epoch, self.epochs, step, len(self.data), loss_real_d, loss_fake_d, loss_real_cls)
                    print "[  Generator  ] epoch: [{:02d}/{}] step: [{:05d}/{}], loss fake: [{:.4f}], class loss: [{:.4f}], reconstruct loss: [{:.4f}]".format(epoch, self.epochs, step, len(self.data), loss_fake_dg, loss_fake_cls, loss_reconst)

                    if step % 20 == 0 and step:

                        summary_loss, summary_images = sess.run([self.summary_loss, self.summary_images], feed_dict)
                        self.summary_writer.add_summary(summary_loss, step + self.length * epoch)
                        self.summary_writer.add_summary(summary_images, step + self.length * epoch)
                        print "Summary [{:d}] added....".format(step)

                    if step % 100 == 0 and step:

                        self.checkpoint_save(sess, step)
                        print "Model {:d} saved......".format(step)

                self.data.reset()




    def test(self):

        with tf.Session(config=self.config) as sess:

            if self.checkpoint_load(sess):
                print "Load from last checkpoint......"
            else:
                raise ValueError('No checkpoint find, Train the model first')

            for idx, (images, labels) in enumerate(self.data):
                labels_list = self.create_labels(labels, self.selected_attrs)

                x_fake_list = []
                for label in labels_list:
                    images_fake = sess.run(self.images_fake, {self.images_real: images, self.labels_trg: labels})
                    x_fake_list.append(torch.from_numpy(images_fake).permute(0, 3, 1, 2))

                x_concat = torch.cat(x_fake_list, dim=3)
                result_path = os.path.join(self.result_dir, '{}-images.jpg'.format(idx))
                save_image(x_concat, result_path, nrow=1, padding=0)
                print('Saved real and fake images into {}...'.format(result_path))

    
    def create_labels(self, c_org, selected_attrs=None):
        """Generate target domain labels for debugging and testing."""

        # Get hair color indices.
        hair_color_indices = []
        for i, attr_name in enumerate(selected_attrs):
            if attr_name in ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair']:
                hair_color_indices.append(i)

        c_trg_list = [c_org.copy()]

        for i in range(len(selected_attrs)):

            c_trg = c_org.copy()
            if i in hair_color_indices:  # Set one hair color to 1 and the rest to 0.
                c_trg[:, i] = 1
                for j in hair_color_indices:
                    if j != i:
                        c_trg[:, j] = 0
            else:
                c_trg[:, i] = (c_trg[:, i] == 0)  # Reverse attribute value.

            c_trg_list.append(c_trg)

        return c_trg_list        



##########  unit test  ##########
if __name__ == '__main__':
    pass