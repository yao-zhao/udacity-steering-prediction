from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

#from six.moves import xrange

import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.training import moving_averages

FLAGS = tf.app.flags.FLAGS

# get variable
def _get_variable(name,
                  shape,
                  initializer,
                  wd=None,
                  trainable=True):
    dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
    with tf.device('/cpu:0'):
        var = tf.get_variable(name,
                              shape=shape,
                              initializer=initializer,
                              dtype=dtype,
                              trainable=trainable)
    if wd is not None:
        weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name=name+'_weight_loss')
        tf.add_to_collection(FLAGS.LOSSES_COLLECTION, weight_decay)
    return var

# single convolution layer
def conv(x, numoutput, ksize=3, stride=1,
         wd=FLAGS.weight_decay, stddev=FLAGS.stddev):
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
def bn(x, moving_average_decay=FLAGS.bn_moving_average_decay,
       bn_epsilon=FLAGS.bn_epsilon):
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
    tf.add_to_collection(FLAGS.UPDATE_OPS_COLLECTION, update_moving_mean)
    tf.add_to_collection(FLAGS.UPDATE_OPS_COLLECTION, update_moving_variance)
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

# global ave pool
def global_ave_pool(x):
    x = tf.reduce_mean(x, reduction_indices=[1, 2], name="global_avg_pool")

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

## softmax loss
#def softmax_loss(logits, labels):
#    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels)
#    cross_entropy_mean = tf.reduce_mean(cross_entropy)
# 
#    regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
#
#    loss_ = tf.add_n([cross_entropy_mean] + regularization_losses)
##    tf.scalar_summary('loss', loss_)
#    return loss_
    
# euclidean loss
def mean_squared_loss(outputs, labels):
    loss = tf.reduce_mean(tf.squared_difference(outputs, labels))
    tf.add_to_collection(FLAGS.LOSSES_COLLECTION, loss)
    return loss
    
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
def stack(x, func, numoutputs):
    for i, numoutput in zip(range(len(numoutputs)), numoutputs):
        with tf.variable_scope(chr(97+i)):
            x=func(x, numoutput)
    return x
    