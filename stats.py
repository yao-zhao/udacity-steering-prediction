import csv
import numpy as np
import os as os
import matplotlib.pyplot as plt
import sys
sys.path.append('/home/yz/caffe3/python')
import sort_human

allfile='data/all.txt'
dup_labels = []

with open(allfile,'r') as csvfile:
    csvreader = csv.reader(csvfile, delimiter=',')
    for row in csvreader:
        dup_labels.append(float(row[1]))

dup_labels = np.asarray(dup_labels)

print(np.sum(np.abs(dup_labels)>0.2), 'of',dup_labels.size)