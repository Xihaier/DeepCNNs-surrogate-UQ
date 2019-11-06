import tensorflow as tf


'''
initialization for weights
1. variance_scaling for CNN
2. weight_xavier for FNN
'''
def weight_vs(shape):
    initializer = tf.contrib.layers.variance_scaling_initializer()
    return tf.Variable(initializer(shape))


def weight_xavier(shape):
    initializer = tf.contrib.layers.xavier_initializer()
    return tf.Variable(initializer(shape))


'''
define operators
1. con2d
2. conv2d_T
3. transition_layer
'''
def conv2d(x, in_features, out_features, kernel_size, stride, pad1, pad2):
    y = tf.pad(x, pad1, "CONSTANT")
    y = tf.nn.conv2d(
        input = y, 
        filter = weight_vs([kernel_size, kernel_size, in_features, out_features]), 
        strides = stride, 
        padding = pad2)
    return y


def conv2d_T(x, in_features, out_features, kernel_size, stride, pad, out_h, out_w):
    batch_size = tf.shape(x)[0]
    y = tf.nn.conv2d_transpose(
        value = x, 
        filter = weight_vs([kernel_size, kernel_size, out_features, in_features]), 
        output_shape = tf.stack([batch_size, out_h, out_w, out_features]), 
        strides = stride, 
        padding = pad)
    return y


def transition_layer(x, in_features, out_features, kernel_size, stride, pad1, pad2, is_training, keep_prob):
    y = tf.contrib.layers.batch_norm(
        inputs = x, 
        scale = True, 
        is_training = is_training, 
        updates_collections = None)
    y = tf.nn.relu(y)
    y = conv2d(y, in_features, out_features, kernel_size, stride, pad1, pad2)
    y = tf.nn.dropout(y, keep_prob)
    return y


def upsampling(x, in_features, out_features, kernel_size, stride, pad1, pad2, out_h, out_w, is_training):
    y = tf.image.resize_bicubic(
        images = x,
        size = [out_h, out_w])
    y = tf.contrib.layers.batch_norm(
        inputs = y, 
        scale = True, 
        is_training = is_training, 
        updates_collections = None)
    y = tf.nn.relu(y)
    y = conv2d(y, in_features, out_features, kernel_size, stride, pad1, pad2)
    return y


'''
define blocks
1. dense_block
'''
def dense_block(x, layers, in_features, growth, kernel_size, stride, pad1, pad2, is_training, keep_prob):
    y = x
    features = in_features
    for idx in range(layers):
        tmp = transition_layer(y, features, growth, kernel_size, stride, pad1, pad2, is_training, keep_prob)
        y = tf.concat((y, tmp), axis=3)
        features += growth
    return y, features

