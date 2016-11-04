import csv
import numpy as np
import os as os
import matplotlib.pyplot as plt

image_list_file='data/train/interpolated.csv'
datapath='data/train'
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
    
def write_to_file(datafile='data/train.txt'):
    dup_filenames, dup_labels, dup_frameids =  get_duplicate_labels()
    with open(datafile,'+w') as file:
        for filename, label, frameid in\
                zip(dup_filenames, dup_labels, dup_frameids):
            file.write('%s,%f,%d' % (filename, label, frameid))
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

