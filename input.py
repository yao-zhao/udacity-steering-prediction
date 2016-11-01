# reading the input from files
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from six.moves import xrange

import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
from tensorflow.python.training import queue_runner

# parse the steering csv file
def _read_steering_csv(image_list_file='data/train/interpolated.csv',
                       datapath='data/train'):
    f = open(image_list_file, 'r')
    filenames = []
    frameids = []
    labels = []
    lineid = 0
    for line in f:
        lineid += 1
        if lineid == 1:
            continue
        index, stamp, width, height, frameid, filename, label =\
            line.split(',')[:7]
        filenames.append(os.path.join(datapath, filename))
        labels.append(float(label))
        if frameid == 'left_camera':
            frameids.append(0)
        elif frameid == 'center_camera':
            frameids.append(1)
        elif frameid == 'right_camera':
            frameids.append(2)
    return filenames, labels, frameids

# split list to tensor
def _split_list_to_tensor(image_list, label_list, frameids, loc):
    sub_image_list = []
    sub_label_list = []
    for image, label, frameid in zip(image_list, label_list, frameids):
        if loc == frameid:
            sub_image_list.append(image)
            sub_label_list.append(label)
    return ops.convert_to_tensor(sub_image_list, dtype=dtypes.string),\
        ops.convert_to_tensor(sub_label_list, dtype=dtypes.float32)

# read image from the disk
def _read_images_from_disk(input_queue):
    label = input_queue[1]
    file_contents = tf.read_file(input_queue[0])
    example = tf.image.decode_jpeg(file_contents, channels=3)
    return example, label

# generate batch
def _generate_batch(image, label, batch_size, min_after_dequeue=10000,
                    num_preprocess_threads=8):
    capacity = min_after_dequeue + (num_preprocess_threads+1) * batch_size
    image_batch, label_batch = tf.train.shuffle_batch(
        [image, label],
        batch_size=batch_size,
        num_threads=num_preprocess_threads,
        capacity=capacity,
        min_after_dequeue=min_after_dequeue)
    return image_batch, label_batch

# data augmentation
def _preprocess_image(image):
    image  =tf.image.resize_images(image, [240, 320, 3])
    #  distorted_image = tf.random_crop(reshaped_image, [height, width, 3])
    image = tf.image.random_brightness(image, max_delta=63)
    image = tf.image.random_contrast(image, lower=0.2, upper=1.8)
    # Subtract off the mean and divide by the variance of the pixels.
    image = tf.image.per_image_whitening(image)

# input pipline
def input_pipline(filenames, batch_size, num_epochs=None,
                  image_list_file='data/train/interpolated.csv',
                  datapath='data/train'):
    image_list, label_list, frameid_list = \
        _read_steering_csv(image_list_file=image_list_file,
                           datapath=datapath)
    center_images, cent_labels = _split_list_to_tensor(image_list,
                                                  label_list, frameid_list, 1)
    input_queue = tf.train.slice_input_producer([center_images, cent_labels],
                                                num_epochs=None,
                                                shuffle=True)
    image, label = _read_images_from_disk(input_queue)
    image = _preprocess_image(image)
    return _generate_batch(image, label, batch_size)

