# -*- coding:utf-8 -*-
# author:peng
# Date：2023/7/3 19:36
from math import ceil

import torch
from itertools import product as product


class Anchors(object):
    def __init__(self, cfg, image_size=None):
        super(Anchors, self).__init__()
        self.min_sizes = cfg['min_sizes']
        self.steps = cfg['steps']
        self.clip = cfg['clip']
        # ---------------------------#
        #   图片的尺寸
        # ---------------------------#
        self.image_size = image_size
        # ---------------------------#
        #   三个有效特征层高和宽
        # ---------------------------#
        self.feature_maps = [[ceil(self.image_size[0] / step), ceil(self.image_size[1] / step)] for step in self.steps]

    def get_anchors(self):
        anchors = []
        # feature_maps : [[80,80], [40,40], [20,20]]
        for k, f in enumerate(self.feature_maps):
            # min_sizes : [[16, 32], [64, 128], [256, 512]]
            min_sizes = self.min_sizes[k]
            # -----------------------------------------#
            #   对特征层的高和宽进行循环迭代
            # -----------------------------------------#
            for i, j in product(range(f[0]), range(f[1])):  # 80*80*2+40*40*2+20*20*2
                for min_size in min_sizes:
                    s_kx = min_size / self.image_size[1]
                    s_ky = min_size / self.image_size[0]
                    dense_cx = [x * self.steps[k] / self.image_size[1] for x in [j + 0.5]]
                    dense_cy = [y * self.steps[k] / self.image_size[0] for y in [i + 0.5]]
                    for cy, cx in product(dense_cy, dense_cx):
                        anchors += [cx, cy, s_kx, s_ky]

        output = torch.Tensor(anchors).view(-1, 4)
        if self.clip:
            output.clamp_(max=1, min=0)
        return output
