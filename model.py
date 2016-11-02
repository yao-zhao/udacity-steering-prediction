
#from six.moves import xrange

import input_steering
import common
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS
# basics
tf.app.flags.DEFINE_integer('batch_size', 16,
                            """Number of images to process in a batch.""")
# naming
tf.app.flags.DEFINE_string('UPDATE_OPS_COLLECTION', 'update_ops',
                          """ collection of ops to be updated""")   
tf.app.flags.DEFINE_string('LOSSES_COLLECTION', 'losses',
                          """ collection of ops to be updated""")
# training
tf.app.flags.DEFINE_integer('num_epochs_per_decay', 4,
                          """number of epochs per decay""")
tf.app.flags.DEFINE_float('initial_learning_rate', 0.01,
                          """initial learning rate""")
tf.app.flags.DEFINE_float('learning_rate_decay', 0.1,
                          """decay factor of learning rate""")
tf.app.flags.DEFINE_float('momentum', 0.9,
                          """momentum of optimization""")

# get input train
def get_train_input():
    images, labels = input_steering.input_pipline(FLAGS.batch_size)
    dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
    images = tf.cast(images, dtype)
    labels = tf.cast(labels, dtype)
    return images, labels

# inference of resnet
def inference_small(images):
    with tf.variable_scope('1'):
        conv1 = common.conv(images, 64, ksize=7, stride=2)
        conv1 = common.bn(conv1)
        pool1 = common.max_pool(conv1)
    with tf.variable_scope('2'):
        stack2 = common.stack(pool1, common.res_block, [64, 64])
        pool2 = common.max_pool(stack2)
    with tf.variable_scope('3'):
        stack3 = common.stack(pool2, common.res_block, [128, 128])
        pool3 = common.max_pool(stack3)
    with tf.variable_scope('4'):
        stack4 = common.stack(pool3, common.res_block, [256, 256, 256])
        pool4 = common.max_pool(stack4)
    with tf.variable_scope('5'):
        stack5 = common.stack(pool4, common.res_block, [512, 512])                                                     
        pool5 = common.global_ave_pool(stack5)
    with tf.variable_scope('fc'):
        fc = common.fc(pool5, 1)
    return fc

# inference of resnet
def inference_resnet(images):
    with tf.variable_scope('1'):
        conv1 = common.conv(images, 64, ksize=7, stride=2)
        conv1 = common.bn(conv1)
        pool1 = common.max_pool(conv1)
    with tf.variable_scope('2'):
        stack2 = common.res_stack(pool1, [256, 256, 256], pool=False)
    with tf.variable_scope('3'):
        stack3 = common.res_stack(stack2, [512, 512, 512, 512])
    with tf.variable_scope('4'):
        stack4 = common.res_stack(stack3, [1024, 1024, 1024,
                                           1024, 1024, 1024])
    with tf.variable_scope('5'):
        stack5 = common.res_stack(stack4, [2048, 2048, 2048])
        pool5 = common.global_ave_pool(stack5)
    with tf.variable_scope('fc'):
        fc = common.fc(pool5, 1)
    return fc

# loss 
def loss(outputs, labels):
    common.mean_squared_loss(outputs, labels)
    return tf.add_n(tf.get_collection(FLAGS.LOSSES_COLLECTION))
    
# train
def train_op(total_loss, global_step):
    with tf.variable_scope('train_op'):
        # learn rate
        num_batches_per_epoch = \
            FLAGS.num_examples_train/FLAGS.batch_size
        decay_steps = int(num_batches_per_epoch * FLAGS.num_epochs_per_decay)
        lr = tf.train.exponential_decay(FLAGS.initial_learning_rate,
                                        global_step,
                                        decay_steps,
                                        FLAGS.learning_rate_decay,
                                        staircase=False)
        # summary
        tf.scalar_summary('learning_rate', lr)
        tf.scalar_summary('total_loss', total_loss)
        # optimization
        opt = tf.train.MomentumOptimizer(lr, FLAGS.momentum, use_nesterov=True)
        grads = opt.compute_gradients(total_loss)
        apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
        # batch norm update
        batchnorm_updates = tf.get_collection(FLAGS.UPDATE_OPS_COLLECTION)
        batchnorm_updates_op = tf.group(*batchnorm_updates)
        # output no op
        with tf.control_dependencies([apply_gradient_op, batchnorm_updates_op]):
            train_op = tf.no_op(name='train')
        return train_op