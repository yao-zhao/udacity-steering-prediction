import csv
import matplotlib.pyplot as plt
import numpy as np
import os

import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('datafile',
                           'data/train.txt',
                           """processed duplicated of train data""")   
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
    # image = tf.image.random_brightness(image, max_delta=63)
    # image = tf.image.random_contrast(image, lower=0.2, upper=1.8)
    # image = tf.image.per_image_whitening(image)
    return image

# process label
def _preprocess_label(label):
    # return label/10
    return label

# image net preprocess on top of data augmentation
def _imagenet_preprocess(rgb):
    """Changes RGB [0,1] valued image to BGR [0,255] with mean subtracted."""
    # red, green, blue = tf.split(2, 3, rgb * 255.0)
    red, green, blue = tf.split(2, 3, rgb)
    bgr = tf.concat(2, [blue, green, red])
    bgr -= IMAGENET_MEAN_BGR
    return bgr


# get all labels in different training set
def _get_all_labels():
    filenames = []
    frameids = []
    labels = []
    for image_list_file, datapath in zip(['data/train/interpolated.csv',
                                          'data/train1/interpolated.csv',
                                          'data/train2/interpolated.csv'],
                                         ['data/train',
                                          'data/train1',
                                          'data/train2']):
        _filename, _label, _frameid =\
            _read_steering_csv(image_list_file, datapath)
        filenames.extend(_filename)
        labels.extend(_label)
        frameids.extend(_frameid)
    return filenames, labels, frameids

# get histogram of all labels
def _get_hist(labels, numbins=50):
    plt.figure(figsize=[8, 6])
    n, bins, patches = plt.hist(labels, numbins, facecolor='green', alpha=0.75)
    plt.xlabel('steering angle')
    plt.ylabel('Probability')
    plt.gca().set_yscale('log')
    plt.axis([-10, 10, 0, np.max(n)*1.01])
    #plt.grid(True)
    plt.savefig('label_hist.png')
    return n, bins, patches
    
# calculate repeat needed for equal sampling
def _get_repeat(n, maxrepeat=50):
    repeat = np.ceil(np.sqrt(np.max(n)/n)).astype(np.int)
    repeat[repeat>maxrepeat] = maxrepeat
    return repeat

# duplicate label list
def _duplicate_labels(filenames, labels, frameids, repeats, bins):
    dup_filenames = []
    dup_labels = []
    dup_frameids = []
    for filename, label, frameid in zip(filenames, labels, frameids):
        for bin_start, bin_end, repeat in zip(bins[:-1], bins[1:], repeats):
            if label >= bin_start and label < bin_end:
                for _ in range(repeat):
                    dup_filenames.append(filename)
                    dup_labels.append(label)
                    dup_frameids.append(frameid)
    return dup_filenames, dup_labels, dup_frameids

def get_duplicate_labels():
    filenames, labels, frameids = _get_all_labels()
    n, bins, patches = _get_hist(labels)
    repeats =_get_repeat(n)
    dup_filenames, dup_labels, dup_frameids = \
        _duplicate_labels(filenames, labels, frameids, repeats, bins)
    return dup_filenames, dup_labels, dup_frameids

def write_to_file(datafile='data/train.txt', caffefile='data/train_caffe.txt'):
    dup_filenames, dup_labels, dup_frameids =  get_duplicate_labels()
    with open(datafile,'+w') as file, open(caffefile, '+w') as caffefile:
        for filename, label, frameid in\
                zip(dup_filenames, dup_labels, dup_frameids):
            file.write('%s,%f,%d\n' % (filename, label, frameid))
            if frameid == 1:
              caffefile.write('%s %f\n' % (filename, label))
    return dup_filenames, dup_labels, dup_frameids

def read_from_file(datafile='data/train.txt'):
    dup_filenames = []
    dup_labels = []
    dup_frameids = []
    with open(datafile,'r') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',')
        for row in csvreader:
            dup_filenames.append(row[0])
            dup_labels.append(row[1])
            dup_frameids.append(row[2])
    return dup_filenames, dup_labels, dup_frameids

# input pipline
def input_pipline(batch_size, num_epochs=None,
                  force_create=False, datafile=FLAGS.datafile):
    # image_list, label_list, frameid_list = \
    #     _read_steering_csv(image_list_file=image_list_file,
    #                        datapath=datapath)
    if os.path.exists(datafile) and not force_create:
        filenames, labels, frameids = read_from_file(datafile=datafile)
    else:
        print(datafile+' not exist not force create is on, start writing')
        filenames, labels, frameids = write_to_file(datafile=datafile)
    center_image_names, cent_labels = _split_list_to_tensor(filenames,
                                                  labels, frameids, 1)
    print('total number of examples: ',len(labels)/3)
    input_queue = tf.train.slice_input_producer([center_image_names,
                                                 cent_labels],
                                                num_epochs=None,
                                                shuffle=True)
    image, label = _read_images_from_disk(input_queue)
    image = _preprocess_image(image)
    image = _imagenet_preprocess(image)
    label = _preprocess_label(label)
    return _generate_batch(image, label, batch_size)

