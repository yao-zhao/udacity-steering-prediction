from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import xrange  # pylint: disable=redefined-builtin

import model

from datetime import datetime
import numpy as np
import os
import time

import tensorflow as tf

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('logdir', 'log/',
                            """Directory to keep log""")
tf.app.flags.DEFINE_string('resnet_param', 'resnet_data/ResNet-50-transfer.ckpt',
                            """resnet parameters to be transfered""")
tf.app.flags.DEFINE_boolean('minimal_summaries', False,
                            """whether to log everything""")
tf.app.flags.DEFINE_integer('max_epoch', 12,
                            """how many epochs to run""")
def main(argv=None):  # pylint: disable=unused-argument
    pass

if __name__ == '__main__':
    tf.app.run()
    


with tf.Graph().as_default():
    # set flag to training
    FLAGS.batch_size=8
    FLAGS.is_training=True
    FLAGS.minimal_summaries=True
    FLAGS.initial_learning_rate=1e-4
    FLAGS.stddev=5e-2
    FLAGS.weight_decay=5e-5
    # global step
    global_step = tf.Variable(0, trainable=False)
    # get training batch
    images, labels = model.get_train_input()
    # inference
    outputs = model.inference_resnet(images)
    # calculate total loss
    loss = model.loss(outputs, labels)
    # train operation
    train_op = model.train_op(loss, global_step)
    # initialize
    init = tf.initialize_all_variables()
    # Start running operations on the Graph.
    config = tf.ConfigProto(log_device_placement=False)
    config.gpu_options.allow_growth = True
    sess = tf.InteractiveSession(config=config)
    sess.run(init)
    # resnet saver
    # saver_resnet = tf.train.Saver(tf.trainable_variables())
    saver_resnet = tf.train.Saver([v for v in tf.trainable_variables()
                                   if not "fc" in v.name])
    saver_resnet.restore(sess, FLAGS.resnet_param)
    # start queue runner
    tf.train.start_queue_runners(sess=sess)
    # pass once
    loss_value = sess.run(loss)
    outputs.eval()

    #
    max_iter = int(FLAGS.max_epoch*
                   FLAGS.num_examples_train/FLAGS.batch_size)
    print('total iteration:', str(max_iter))
    for step in xrange(max_iter):
          start_time = time.time()
          _, loss_value = sess.run([train_op, loss])
          duration = time.time() - start_time
          assert not np.isnan(loss_value), 'Model diverged with loss = NaN'
          
          if step % 10 == 0:
              num_examples_per_step = FLAGS.batch_size
              examples_per_sec = num_examples_per_step / duration
              sec_per_batch = float(duration)
              format_str = ('%s: step %d, loss = %.2f'
                            ' (%.1f examples/sec; %.3f sec/batch)')
              print (format_str % (datetime.now(), step, loss_value,
                                   examples_per_sec, sec_per_batch))
    
          if step % 200 == 0:
              summary_str = sess.run(summary_op)
              summary_writer.add_summary(summary_str, step)
    
          # Save the model checkpoint periodically.
          if step % 1000 == 0 or (step + 1) == max_iter:
              checkpoint_path = os.path.join(FLAGS.logdir, 'model.ckpt')
              saver.save(sess, checkpoint_path, global_step=step)
                                      
