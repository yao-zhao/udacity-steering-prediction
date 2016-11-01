from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import model
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

def train():
    with tf.Graph().as_default():
        # global step
        global_step = tf.Variable(0, trainable=False)
        # get training batch
        images, labels = model.get_train_input()
        # inference
        outputs = model.inference(images)
        # calculate total loss
        loss = model.loss(outputs, labels)
        # train operation
        train_op = model.train(loss, global_step)
        # saver
        saver = tf.train.Saver(tf.all_variables())
        # summarize
        summary_op = tf.merge_all_summaries()
        # initialize
#        init = tf.initialize_all_variables()
#        # Start running operations on the Graph.
#        config = tf.ConfigProto(log_device_placement=\
#                                FLAGS.log_device_placement)
#        config.gpu_options.allow_growth = True
#        sess = tf.Session(config=config)
#        sess.run(init)
        
def main(argv=None):  # pylint: disable=unused-argument
  train()

if __name__ == '__main__':
  tf.app.run()
