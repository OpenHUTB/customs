# -*- coding:utf-8 -*-
# author:peng
# Date：2023/7/3 19:20
import torch

cfg_mnet = {
    # ----------------------------------------------------------------------#
    #   anchor构建
    # ----------------------------------------------------------------------#
    'min_sizes'         : [[16, 32], [64, 128], [256, 512]],
    'steps'             : [8, 16, 32],
    'variance'          : [0.1, 0.2],
    'clip'              : False,
    # ----------------------------------------------------------------------#
    #   图像检测中会将图片resize的大小
    # ----------------------------------------------------------------------#
    'test_image_size'   : 640,
    # ----------------------------------------------------------------------#
    #   retinaface中只有得分大于置信度的预测框会被保留下来
    # ----------------------------------------------------------------------#
    "confidence"        : 0.5,
    # ----------------------------------------------------------------------#
    #   retinaface中非极大抑制所用到的nms_iou大小
    # ----------------------------------------------------------------------#
    "nms_iou"           : 0.3,
    # ----------------------------------------------------------------------#
    #   arcface模型不同人脸的阈值
    # ----------------------------------------------------------------------#
    "facenet_threhold"  : 27.0,
    # ----------------------------------------------------------------------#
    #   faiss索引不同人脸的阈值
    # ----------------------------------------------------------------------#
    "faiss distance"    : 750,
    'device'            : 'cuda:0' if torch.cuda.is_available() else 'cpu'
}