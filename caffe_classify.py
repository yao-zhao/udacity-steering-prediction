import csv
import numpy as np
import os as os
import matplotlib.pyplot as plt
import sys
sys.path.append('/home/yz/caffe3/python')
import sort_human
import caffe

def _get_testnames(testfile='data/test_caffe.txt'):
    with open(testfile, 'r') as file:
        filenames = []
        reader = csv.reader(file, delimiter=' ')
        for row in reader:
            name = row[0].split('/')
            filenames.append(name[2])
    return filenames

def inference(\
    caffe_model = 'caffe_model/test.prototxt',\
    caffe_weights = 'caffe_model/resnet_iter_12000.caffemodel'):
    filenames = _get_testnames()
    numfiles = len(filenames)
    caffe.set_mode_gpu()
    caffe.set_device(1)
    net = None
    net = caffe.Net(caffe_model, caffe_weights, caffe.TEST)
    output = net.blobs['fc1'].data
    batchsize = output.shape[0]
    numbatch = int(np.floor(numfiles/int(batchsize))+1)
    labels = np.zeros((numbatch*batchsize,1))
    for ibatch in range(int(numbatch)):
        if ibatch % 100 == 0:
            print(ibatch, ' out of ', numbatch, ' batches')
        net.forward()
        output = net.blobs['fc1'].data    
        labels[ibatch*batchsize:(ibatch+1)*batchsize] = output
    labels = labels[0:numfiles]
    return labels

def write_test(labels, testfile='data/result.txt'):
    with open(testfile, 'w+') as file:
        filenames = _get_testnames()
        file.write('frame_id,steering_angle\n')
        for filename, label in zip(filenames, labels):
            file.write('%s,%f\n' % (filename, label))
            

labels = inference()
write_test(labels)