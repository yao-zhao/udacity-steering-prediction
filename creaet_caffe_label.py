import csv
import numpy as np
import os as os
import matplotlib.pyplot as plt

def create_caffe_train_val(allfile='data/all.txt',
                           trainfile='data/train_caffe.txt',
                           valfile='data/val_caffe.txt',
                           ratio=10):
    dup_filenames = []
    dup_labels = []
    dup_frameids = []

    with open(allfile,'r') as csvfile, \
            open(trainfile,'w+') as train, \
            open(valfile, 'w+') as val:
        csvreader = csv.reader(csvfile, delimiter=',')
        counter = 0
        for row in csvreader:
            if int(row[2]) == 1:
                counter += 1
                if int(counter/5000) % ratio == 0:
                    val.write('%s %s\n' % (row[0], row[1]))
                else:
                    train.write('%s %s\n' % (row[0], row[1]))

create_caffe_train_val()