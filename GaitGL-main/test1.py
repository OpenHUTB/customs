from datetime import datetime
import numpy as np
import argparse

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

def evaluation1(data, config):
    dataset = config['dataset'].split('-')[0]
    feature, view, label, feature1, label1 = data
    # label = np.array(label)
    view_list = list(set(view))
    view_list.sort()
    view_num = len(view_list)
    sample_num = len(feature)
    identify = list()
    name = np.str("")
    for i in range(len(label1)):
        min = 10000
        for j in range(len(label)):
            dist = cuda_dist(feature1[i][np.newaxis, :], feature[j][np.newaxis, :])
            if(dist.item()<min):
                min = dist
                name = label[j]
        identify.append(name)
    return identify,label1
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

        # Exclude identical-view cases

        m = initialization(conf, test=opt.cache)[0]

        # load model checkpoint of iteration opt.iter
        print('Loading the model of iteration %d...' % iter)
        m.load(iter)
        print('Transforming...')
        time = datetime.now()
        test = m.transform('test', opt.batch_size)
        print('Evaluating...')
        acc,label = evaluation1(test, conf['data'])
        for i in range(len(acc)):
            print(label[i],"---->",acc[i])
        print('Evaluation complete. Cost:', datetime.now() - time)

