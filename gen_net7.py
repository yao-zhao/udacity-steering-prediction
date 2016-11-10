import numpy as np
import os
import argparse
import sys
caffe_path = '/home/yz/caffe3/'
sys.path.append(caffe_path+'python')
import caffe
import build_net as b
import os
os.environ["GLOG_minloglevel"] = "2"

def net7(n):
    n.add_image(batch_size = 512, test_batch_size = 32,
        transformer_dict = dict(mirror = False,
        mean_value=128, scale=0.00390625,
        contrast_jitter_range=0.3
        ),
        label_scale=1,
        root_folder='', is_color=True, shuffle=True, test_shuffle=True,
        source_train='train_caffe.txt', source_test='val_caffe.txt',
        height = 192, width = 256)
    bins = [0.01, 0.03, 0.05, 0.075, 0.1, 0.15,
            0.2, 0.3, 0.4, 0.5, 0.7,
            1, 2, 3, 4, 5, 6]
    reversebins = list(bins)
    reversebins.reverse()
    bins = [-x for x in reversebins]+bins
    print('bins:',bins)
    n.add_conv(32, kernel_size=7, stride=2, pad=3, bias_term=True)
    for numoutput in [32, 64, 128, 256, 512]:
        n.add_conv(numoutput, bias_term=True)
        n.add_conv(numoutput, stride=2, bias_term=True)
    n.add_fc(len(bins)+1)
    n.add_softmax_decay(bins, decay_rate = 2)
    n.add_solver_sdg(test_interval = 100, test_iter = 16,
                max_iter = 4e4, base_lr = 0.001, momentum = 0.9,
                weight_decay = 1e-5, gamma = 0.1, stepsize = 1e4,
                display = 10, snapshot = 5e3, iter_size = 1)

net = b.BuildNet(net7, name = 'net7', caffe_path = caffe_path)
net.save()