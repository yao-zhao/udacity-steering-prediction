
import os

import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('train_file',
                           'data/train/interpolated.csv',
                           """image list of training set""")   
tf.app.flags.DEFINE_string('train_dir', 'data/train',
                          """train directory""")  
tf.app.flags.DEFINE_string('val_file',
                           'data/train/interpolated.csv',
                           """image list of training set""")   
tf.app.flags.DEFINE_string('val_dir', 'data/train',
                          """train directory""")  
tf.app.flags.DEFINE_string('test_file',
                           'data/train/interpolated.csv',
                           """image list of training set""")   
tf.app.flags.DEFINE_string('test_dir', 'data/train',
                          """train directory""")
tf.app.flags.DEFINE_integer('num_examples_train', 45000,
                          """number of examples per epoch in training""")

IMAGENET_MEAN_BGR = [103.062623801, 115.902882574, 123.151630838, ]

# parse the steering csv file
def _read_steering_csv(image_list_file, datapath):
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
def _generate_batch(image, label, batch_size, min_after_dequeue=5000,
                    num_preprocess_threads=6):
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
    image  =tf.image.resize_images(image, [180, 240])
    image = tf.random_crop(image, [160, 192, 3])
    image = tf.image.random_brightness(image, max_delta=63)
    image = tf.image.random_contrast(image, lower=0.2, upper=1.8)
    image = tf.image.per_image_whitening(image)
    return image

# image net preprocess on top of data augmentation
def _imagenet_preprocess(rgb):
    """Changes RGB [0,1] valued image to BGR [0,255] with mean subtracted."""
    red, green, blue = tf.split(2, 3, rgb * 255.0)
    bgr = tf.concat(2, [blue, green, red])
    bgr -= IMAGENET_MEAN_BGR
    return bgr

# input pipline
def input_pipline(batch_size, num_epochs=None,
                  image_list_file=FLAGS.train_file,
                  datapath=FLAGS.train_dir):
    image_list, label_list, frameid_list = \
        _read_steering_csv(image_list_file=image_list_file,
                           datapath=datapath)
    center_image_names, cent_labels = _split_list_to_tensor(image_list,
                                                  label_list, frameid_list, 1)
    input_queue = tf.train.slice_input_producer([center_image_names,
                                                 cent_labels],
                                                num_epochs=None,
                                                shuffle=True)
    image, label = _read_images_from_disk(input_queue)
    image = _preprocess_image(image)
    image = _imagenet_preprocess(image)
    return _generate_batch(image, label, batch_size)

