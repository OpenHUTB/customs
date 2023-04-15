from datetime import datetime
import numpy as np
import argparse
import os
from model.initialization import initialization
from model.utils import evaluation
from config import conf
import torch
import torch.nn.functional as F

def boolean_string(s):
    if s.upper() not in {'FALSE', 'TRUE'}:
        raise ValueError('Not a valid boolean string')
    return s.upper() == 'TRUE'


def de_diag(acc, each_angle=False):
    result = np.sum(acc - np.diag(np.diag(acc)), 1) / 10.0
    if not each_angle:
        result = np.mean(result)
    return result


# Exclude identical-view cases
def de_diag13(acc, each_angle=False):
    result = np.sum(acc - np.diag(np.diag(acc)), 1) / 13.0
    if not each_angle:
        result = np.mean(result)
    return result

def cuda_dist(x, y):
    x = torch.from_numpy(x).cuda()
    y = torch.from_numpy(y).cuda()
    dist = torch.sum(x ** 2, 1).unsqueeze(1) + torch.sum(y ** 2, 1).unsqueeze(
        1).transpose(0, 1) - 2 * torch.matmul(x, y.transpose(0, 1))
    dist = torch.sqrt(F.relu(dist))
    return dist

if __name__ == '__main__':

    iterall = [80000]
    for iter in iterall:
        parser = argparse.ArgumentParser(description='Test')

        parser.add_argument('--batch_size', default='1', type=int,
                            help='batch_size: batch size for parallel test. Default: 1')
        parser.add_argument('--cache', default=False, type=boolean_string,
                            help='cache: if set as TRUE all the test data will be loaded at once'
                                 ' before the transforming start. Default: FALSE')
        opt = parser.parse_args()

        print("Initialzing...")
        WORK_PATH = conf['WORK_PATH']
        os.chdir(WORK_PATH)
        os.environ["CUDA_VISIBLE_DEVICES"] = conf["CUDA_VISIBLE_DEVICES"]

        # Exclude identical-view cases
        print('Loading the model of iteration %d...' % iter)
        m.load(iter)
        print('Transforming...')

