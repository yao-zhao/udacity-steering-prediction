import numpy as np

import matplotlib.pyplot as plt

image_list_file='data/train/interpolated.csv'
datapath='data/train'

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

plt.figure(figsize=[8, 6])
n, bins, patches = plt.hist(labels, 50, facecolor='green', alpha=0.75)
plt.xlabel('steering angle')
plt.ylabel('Probability')
plt.axis([-10, 10, 0, np.max(n)*1.01])
#plt.grid(True)

plt.savefig('label_hist.png')