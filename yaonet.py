from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

#from six.moves import xrange

import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.training import moving_averages

FLAGS = tf.app.flags.FLAGS

# Basic model parameters.
tf.app.flags.DEFINE_integer('batch_size', 16,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_string('dtype', 'float32',
                            """Train the model using datatype.""")
tf.app.flags.DEFINE_boolean('is_training', False,
                            """Boolean to decide if the model is training""")

UPDATE_OPS_COLLECTION = 'update_ops'  # must be grouped with training op
LOSSES_COLLECTION = 'losses'




# get variable
def _get_variable(name,
                  shape,
                  initializer,
                  wd=None,
                  dtype='float',
                  trainable=True):
    with tf.device('/cpu:0'):
        var = tf.get_variable(name,
                              shape=shape,
                              initializer=initializer,
                              dtype=dtype,
                              trainable=trainable)
    if wd is not None:
        weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name=name+'_weight_loss')
        tf.add_to_collection(LOSSES_COLLECTION, weight_decay)
    return var

# single convolution layer
def conv(x, numoutput, ksize=3, stride=1, wd=1e-6, stddev=0.1):
    filters_in = x.get_shape()[-1]
    shape = [ksize, ksize, filters_in, numoutput]
    initializer = tf.truncated_normal_initializer(stddev=stddev)
    weights = _get_variable('weights',
                            shape=shape,
                            dtype=FLAGS.dtype,
                            initializer=initializer,
                            weight_decay=wd)
    return tf.nn.conv2d(x, weights, [1, stride, stride, 1], padding='SAME')

# batch normalization
def bn(x, moving_average_decay=0.999, bn_epsilon=0.0001):
    x_shape = x.get_shape()
    depth = x_shape[-1:]
    # averaging axis
    axis = list(range(len(x_shape) - 1))
    # rescale and baise
    beta = _get_variable('beta',
                         depth,
                         initializer=tf.zeros_initializer)
    gamma = _get_variable('gamma',
                          depth,
                          initializer=tf.ones_initializer)
    # save for moving average
    moving_mean = _get_variable('moving_mean',
                                depth,
                                initializer=tf.zeros_initializer,
                                trainable=False)
    moving_variance = _get_variable('moving_variance',
                                    depth,
                                    initializer=tf.ones_initializer,
                                    trainable=False)
    mean, variance = tf.nn.moments(x, axis)
    update_moving_mean = moving_averages.assign_moving_average(\
                         moving_mean, mean, moving_average_decay)
    update_moving_variance = moving_averages.assign_moving_average(\
                             moving_variance, variance, moving_average_decay)
    tf.add_to_collection(UPDATE_OPS_COLLECTION, update_moving_mean)
    tf.add_to_collection(UPDATE_OPS_COLLECTION, update_moving_variance)
    mean, variance = control_flow_ops.cond(
        FLAGS.is_training, lambda: (mean, variance),
        lambda: (moving_mean, moving_variance))
    x = tf.nn.batch_normalization(x, mean, variance, beta, gamma, bn_epsilon)
    return x
    
# activation
def activation(x):
    return tf.nn.relu(x)
    
# max pool
def max_pool(x, ksize=3, stride=2):
    return tf.nn.max_pool(x,
                          ksize=[1, ksize, ksize, 1],
                          strides=[1, stride, stride, 1],
                          padding='SAME')

# fully connected
def fc(x, numoutput, wd=1e-6, stddev=0.1):
    num_units_in = x.get_shape()[1]
    weights_initializer = tf.truncated_normal_initializer(
        stddev=stddev)
    weights = _get_variable('weights',
                            shape=[num_units_in, numoutput],
                            initializer=weights_initializer,
                            weight_decay=wd)
    biases = _get_variable('biases',
                           shape=[numoutput],
                           initializer=tf.zeros_initializer)
    x = tf.nn.xw_plus_b(x, weights, biases)
    return x
    
# resnet block
def res_block(x, numoutput):
    shortcut = x  # branch 1
    internal_numfilters = numoutput/4
    with tf.variable_scope('branch2'):
        with tf.variable_scope('a'):
            x = conv(x, internal_numfilters, ksize=1)
            x = bn(x)
            x = activation(x)
        with tf.variable_scope('b'):
            x = conv(x, internal_numfilters)
            x = bn(x)
            x = activation(x)
        with tf.variable_scope('c'):
            x = conv(x, numoutput, ksize=1)
            x = bn(x)
    with tf.variable_scope('branch1'):
        shortcut = conv(shortcut, numoutput)
        shortcut = bn(shortcut)
    return activation(x + shortcut)
    
# stack of blocks
def stack(x, func, num):
    for i in range(num):
        with tf.variable_scope(chr(97+i)):
            x=func(x)
    return x