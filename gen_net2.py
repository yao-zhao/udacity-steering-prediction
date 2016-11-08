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

def net(n):
    n.add_image(batch_size=512, test_batch_size=512,
        transformer_dict = dict(mirror=False, 
        mean_value=128, scale=0.00390625,
        crop_h=116, crop_w=156, random_crop_train=False
        ),
        label_scale=1,
        root_folder='', is_color=True, shuffle=True, test_shuffle=True,
        source_train='train_caffe.txt', source_test='val_caffe.txt',
        height = 120, width = 160)
    n.add_conv(24, kernel_size=5, stride=2, pad=0, bias_term=True)
    n.add_relu()
    n.add_conv(36, kernel_size=5, stride=2, pad=0, bias_term=True)
    n.add_relu()
    n.add_conv(48, kernel_size=5, stride=2, pad=0, bias_term=True)
    n.add_relu()
    n.add_conv(64, kernel_size=3, stride=1, pad=0, bias_term=True)
    n.add_relu()
    n.add_conv(64, kernel_size=3, stride=1, pad=0, bias_term=True)
    n.add_relu()
    n.add_conv(64, kernel_size=3, stride=1, pad=0, bias_term=True)
    n.add_relu()
    n.add_conv(64, kernel_size=3, stride=1, pad=0, bias_term=True)
    n.add_relu()
    n.add_fc(512)
    n.add_relu()
    n.add_dropout(.5)
    n.add_fc(128)
    n.add_relu()
    n.add_dropout(.375)
    n.add_fc(32)
    n.add_relu()
    n.add_dropout(.25)
    n.add_fc(8)
    n.add_relu()
    n.add_dropout(.125)
    n.add_fc(1)
    n.add_euclidean(loss_weight = 1)

    n.add_solver_sdg(test_interval = 200, test_iter = 16,
                max_iter = 4e4, base_lr = 0.01, momentum = 0.9,
                weight_decay = 1e-4, gamma = 0.1, stepsize = 1e4,
                display = 10, snapshot = 5e3, iter_size = 1)

net = b.BuildNet(net, name = 'net2', caffe_path = caffe_path)
net.save()