import tensorflow as tf
from tensorflow.contrib import slim
import util


def leaky_relu(x, leak=0.2):
    return tf.maximum(x, leak*x)

def res_block(x, kernel_size=3, stride=1, padding='SAME', activation_fn=None, weights_initializer=tf.contrib.layers.xavier_initializer(), biases_initializer=None, trainable=True, scope=None):
    _, _, _, C = x.shape.as_list()
    with tf.variable_scope(scope):
        net = slim.conv2d(x, C, kernel_size=kernel_size, padding=padding, activation_fn=activation_fn, weights_initializer=weights_initializer,
                          biases_initializer=biases_initializer, trainable=trainable, scope='conv1')
        net = tf.contrib.layers.instance_norm(net, activation_fn=activation_fn, scope='ins1')
        net = tf.nn.relu(net)
        net = slim.conv2d(net, C, kernel_size=kernel_size, padding=padding, activation_fn=activation_fn, weights_initializer=weights_initializer,
                          biases_initializer=biases_initializer, trainable=trainable, scope='conv2')
        net = tf.contrib.layers.instance_norm(net, activation_fn=activation_fn, scope='ins2')

        return x + net


def generator(images, labels, num_filters, num_resblks=6, reuse=False):
    """
    images: (None, H, W, C)
    labels: (None, classes)
    num_filters: 64, for example
    num_resblks: 6, for example

    outputs: (None, H, W, C)
    """
    _, H, W, _ = images.shape.as_list()
    labels_exp = tf.to_float(tf.tile(labels[:, None, None, :], [1, H, W, 1])) # (None, image_size, image_size, c_dim)
    inputs = tf.concat([images, labels_exp], axis=3) # (None, image_size, image_size, 3+c_dim)    
    with tf.variable_scope('generator', reuse=reuse):
        with tf.variable_scope('down-sampling'):
            net = slim.conv2d(inputs, num_filters, 7, padding='SAME', stride=1, activation_fn=None, scope='conv1')
            net = tf.contrib.layers.instance_norm(net, activation_fn=None, scope='ins1')
            net = tf.nn.relu(net)
            net = slim.conv2d(net, num_filters*2, 4, padding='SAME', stride=2, activation_fn=None, scope='conv2')
            net = tf.contrib.layers.instance_norm(net, activation_fn=None, scope='ins2')
            net = tf.nn.relu(net)
            net = slim.conv2d(net, num_filters*4, 4, padding='SAME', stride=2, activation_fn=None, scope='conv3')
            net = tf.contrib.layers.instance_norm(net, activation_fn=None, scope='ins3')
            net = tf.nn.relu(net)


        with tf.variable_scope('bottleneck'):
            # with slim.arg_scope([res_block], kernel_size=3, stride=1, padding='SAME', activation_fn=None, weights_initializer=tf.contrib.layers.xavier_initializer(), biases_initializer=None, trainable=True):
            for i in range(1, num_resblks+1):
                net = res_block(net, scope='res{}'.format(i))


        with tf.variable_scope('up-sampling'):
            net = slim.conv2d_transpose(net, num_filters*2, 4, stride=2, padding='SAME', activation_fn=None, scope='deconv1')
            net = tf.contrib.layers.instance_norm(net, activation_fn=None, scope='ins1')
            net = tf.nn.relu(net)
            net = slim.conv2d_transpose(net, num_filters, 4, stride=2, padding='SAME', activation_fn=None, scope='deconv2')
            net = tf.contrib.layers.instance_norm(net, activation_fn=None, scope='ins2')
            net = tf.nn.relu(net)
            net = slim.conv2d(net, 3, 7, stride=1, padding='SAME', activation_fn=None, scope='deconv3')
            net = tf.contrib.layers.instance_norm(net, activation_fn=None, scope='ins3')
            net = tf.tanh(net)

        return net




def discriminator(inputs, num_filters, num_classes, reuse=False):
    """
    inputs: (None, H, W, C)
    num_filters: 64, for example
    num_classes: 10, for example
    num_convs: 6, for example

    logits: (None, H/64, W/64, 1)
    logits_class: (None, 1, 1, num_classes)
    """
    with tf.variable_scope('discriminator', reuse=reuse):
        with slim.arg_scope([slim.conv2d], stride=2, padding='SAME', activation_fn=leaky_relu, weights_initializer=tf.contrib.layers.xavier_initializer()):
            net = slim.conv2d(inputs, num_filters, 4, scope='conv1')
            net = slim.conv2d(net, num_filters*2, 4, scope='conv2')
            net = slim.conv2d(net, num_filters*4, 4, scope='conv3')
            net = slim.conv2d(net, num_filters*8, 4, scope='conv4')
            net = slim.conv2d(net, num_filters*16, 4, scope='conv5')
            net = slim.conv2d(net, num_filters*32, 4, scope='conv6')
            _, h, w, _ = net.shape.as_list()
            logits = slim.conv2d(net, 1, 3, stride=1, activation_fn=None, scope='conv_logit')
            logits_class = slim.conv2d(net, num_classes, (h, w), stride=1, padding='VALID', activation_fn=None, scope='conv_class')

            return logits, tf.reshape(logits_class, (-1, num_classes))

########## unit test ##########
if __name__ == '__main__':
    images = tf.placeholder(tf.float32, (None, 224, 224, 3), name='x')
    _, _ = discriminator(images, 64, 10, False)
    _ = generator(images, 64, 6, False)
    D_variables, G_variables = tf.trainable_variables('discriminator'), tf.trainable_variables('generator') 
    total_dis, total_gen = util.total_params(D_variables), util.total_params(G_variables) 
    print ('  Total trainable variables for discriminator: {:d}  '.format(total_dis)).center(200, '#')
    for var in D_variables:
        print '{}: {}'.format(var.op.name, var.shape.as_list())
    print ('  Total trainable variables for generator: {:d}  '.format(total_gen)).center(200, '#')
    for var in G_variables:
        print '{}: {}'.format(var.op.name, var.shape.as_list())





