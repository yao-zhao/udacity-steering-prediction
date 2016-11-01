from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

#from six.moves import xrange

import input_steering
import common
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS
# basics
tf.app.flags.DEFINE_integer('batch_size', 16,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_boolean('use_fp16', False,
                            """Train the model using fp16.""")
tf.app.flags.DEFINE_boolean('is_training', False,
                            """Boolean to decide if the model is training""")
# model params
tf.app.flags.DEFINE_float('bn_moving_average_decay', 0.999,
                          """ Batch normalizating movine average decay""")
tf.app.flags.DEFINE_float('bn_epsilon', 0.0001,
                          """ Batch normalizating variance epsilon""")
tf.app.flags.DEFINE_float('weight_decay', 1e-6,
                          """ Default weight decal value for all parameters""")                         
tf.app.flags.DEFINE_float('stddev', 0.1,
                          """ Default initialization std""")
# naming
tf.app.flags.DEFINE_string('UPDATE_OPS_COLLECTION', 'update_ops',
                          """ collection of ops to be updated""")   
tf.app.flags.DEFINE_string('LOSSES_COLLECTION', 'losses',
                          """ collection of ops to be updated""")
# training
tf.app.flags.DEFINE_integer('NUM_EPOCHS_PER_DECAY', 4,
                          """number of epochs per decay""")
tf.app.flags.DEFINE_float('INITIAL_LEARNING_RATE', 0.01,
                          """initial learning rate""")
tf.app.flags.DEFINE_float('LEARNING_RATE_DECAY_FACTOR', 0.1,
                          """decay factor of learning rate""")
tf.app.flags.DEFINE_float('MOMENTUM', 0.9,
                          """momentum of optimization""")
# get input train
def get_input_train():
    images, labels = input_steering.input_pipline(FLAGS.batch_size)
    dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
    images = tf.cast(images, dtype)
    labels = tf.cast(labels, dtype)
    images = tf.random_crop(images, [FLAGS.batch_size, 224, 288, 3])
    #    image_batch = tf.image.random_brightness(image_batch, max_delta=63)
    #    image_batch = tf.image.random_contrast(image_batch, lower=0.2, upper=1.8)
    # Subtract off the mean and divide by the variance of the pixels.
    images = tf.image.per_image_whitening(images)

# inference of res net
def inference(images):
    with tf.variable_scope('1'):
        conv = common.conv(images, 64, ksize=7, stride=2)
        conv = common.bn(conv)
        pool = common.max_pool(conv)
    with tf.variable_scope('2'):
        stack = common.stack(pool, common.res_block, [64, 64, 64])
        pool = common.max_pool(stack)
    with tf.variable_scope('3'):
        stack = common.stack(pool, common.res_block, [128, 128, 128, 128])
        pool = common.max_pool(stack)
    with tf.variable_scope('4'):
        stack = common.stack(pool, common.res_block, [256, 256, 256,
                                                      256, 256, 256])
        pool = common.max_pool(conv)
    with tf.variable_scope('5'):
        stack = common.stack(pool, common.res_block, [512, 512, 512])                                                     
        pool = common.global_ave_pool(stack)

# loss 
def loss(outputs, labels):
    common.mean_squared_loss(outputs, labels)
    return tf.add_n(tf.get_collection(FLAGS.LOSSES_COLLECTION))
    
# train
def train_op(total_loss, global_step):
    # learn rate
    num_batches_per_epoch = \
        FLAGS.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN/FLAGS.batch_size
    decay_steps = int(num_batches_per_epoch * FLAGS.NUM_EPOCHS_PER_DECAY)
    lr = tf.train.exponential_decay(FLAGS.INITIAL_LEARNING_RATE,
                                    global_step,
                                    decay_steps,
                                    FLAGS.LEARNING_RATE_DECAY_FACTOR,
                                    staircase=True)
    # summary
    tf.scalar_summary('learning_rate', lr)
    tf.scalar_summary('total_loss', total_loss)
    # optimization
    opt = tf.train.MomentumOptimizer(lr, FLAGS.MOMENTUM, use_nesterov=True)
    grads = opt.compute_gradients(total_loss)
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
    # batch norm update
    batchnorm_updates = tf.get_collection(FLAGS.UPDATE_OPS_COLLECTION)
    batchnorm_updates_op = tf.group(*batchnorm_updates)
    # output no op
    with tf.control_dependencies([apply_gradient_op, batchnorm_updates_op]):
        train_op = tf.no_op(name='train')
    return train_op