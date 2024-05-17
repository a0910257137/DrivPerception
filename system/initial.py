import tensorflow as tf
import numpy as np
import math
from pprint import pprint
from monitor import logger


class Initialize:

    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.inp_size = self.cfg.resize_size
        self.head_cfg = self.cfg.head
        self.strides = self.head_cfg.strides
        anchors = self.head_cfg.anchors
        self.nc = self.head_cfg.num_classes

        self.na = len(anchors)

    def get_branches(self, layers, name):
        branch = {}
        for layer in layers:
            name = layer.name
            if "Detect" in name:
                branch[name] = layer
        return branch

    def __call__(self, model):
        model.model(tf.constant(0., shape=[1] + self.inp_size + [3]),
                    training=False)
        logger.info(f'Manually initialize model parameters')
        head = model.model.get_layer("head")
        det_branches = self.get_branches(head.layers, "Detect")
        keys = sorted(det_branches.keys())
        for i, key in enumerate(keys):
            layer = det_branches[key]
            kernel_data = layer.get_weights()
            weights, bias = kernel_data
            _, _, in_chs, out_chs = weights.shape
            stdv = 1. / np.sqrt(in_chs)
            vals = np.random.uniform(low=-stdv, high=stdv, size=[out_chs])
            vals = np.reshape(vals, (self.na, -1))
            s = self.strides[i]
            vals[:, 4] += math.log(
                8 / (640 / s)**2)  # obj (8 objects per 640 image)
            vals[:, 5:] += math.log(0.6 / (self.nc - 0.99))
            bias = tf.reshape(vals, [-1])
            layer.set_weights([weights, bias])
        return model
