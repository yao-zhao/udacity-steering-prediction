import csv
import numpy as np
import os as os
import matplotlib.pyplot as plt
import sys
sys.path.append('/home/yz/caffe3/python')
import sort_human

allfile='data/all.txt'
labels = []

with open(allfile,'r') as csvfile:
    csvreader = csv.reader(csvfile, delimiter=',')
    for row in csvreader:
        labels.append(float(row[1]))

labels = np.asarray(labels)

#%%
#bins = np.exp(np.linspace(0, 2, 101))-1
bins = np.asarray([0.01, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7,
                   1, 2, 3, 4, 5, 6, 7])
bins = np.concatenate((-bins[::-1], bins))

bins_c = (bins[:-1]+bins[1:])/2
numbins = len(bins)
counts = np.zeros((numbins-1))
mse = 0
mse_b = 0
for label in labels:
    for lb, ub, ib, cb in zip(bins[:-1], bins[1:], range(numbins-1), bins_c):
        if label > lb and label <= ub:
            counts[ib] += 1
            mse += (label - cb)**2
            mse_b += label**2
            continue
mse /= len(labels)
rmse = np.sqrt(mse)
rmse_b = np.sqrt(mse_b/len(labels))
print(rmse, rmse_b)


#plt.bar(bins_c, counts)
#plt.plot(bins_c, counts)
plt.bar(range(len(counts)), counts)