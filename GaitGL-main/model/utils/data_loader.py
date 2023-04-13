import os
import os.path as osp

import numpy as np

from .data_set import DataSet,DataSet1


def load_data(dataset_path, testset_path, resolution, dataset, pid_num, pid_shuffle, cache=True):
    seq_dir = list()
    view = list()
    label = list()

    for _label in sorted(list(os.listdir(dataset_path))):

        label_path = osp.join(dataset_path, _label)
        for _view in sorted(list(os.listdir(label_path))):
            _seq_dir = osp.join(label_path, _view)
            seqs = os.listdir(_seq_dir)
            if len(seqs) > 14:
                seq_dir.append([_seq_dir])
                label.append(_label)
                view.append(_view)
    print("seq_dir",seq_dir)
    print("label", label)
    print("view", view)
    pid_fname = osp.join('partition', '{}_{}_{}.npy'.format(
        dataset, pid_num, pid_shuffle))
    if not osp.exists(pid_fname):
        pid_list = sorted(list(set(label)))
        if pid_shuffle:
            np.random.shuffle(pid_list)
        # pid_list = [pid_list[0:pid_num], pid_list[pid_num:]]
        os.makedirs('partition', exist_ok=True)
        np.save(pid_fname, pid_list)
    print(pid_fname)
    pid_list = np.load(pid_fname, allow_pickle=True)

    # train_list = pid_list[1]
    test_list = pid_list
    print("test_list", test_list)

    seq_dir1 = list()
    label1 = list()
    for _label in sorted(list(os.listdir(testset_path))):

        label_path = osp.join(testset_path, _label)
        seqs = os.listdir(label_path)
        print(label_path)
        if len(seqs) > 14:
            seq_dir1.append([label_path])
            label1.append(_label)
    print("seq_dir1",seq_dir1)
    print("label1",label1)
    use_list = label1
    test_source = DataSet(
        [seq_dir[i] for i, l in enumerate(label) if l in test_list],
        [label[i] for i, l in enumerate(label) if l in test_list],
        [view[i] for i, l in enumerate(label) if l in test_list],
        cache, resolution, cut=True)
    use_source = DataSet1(
        [seq_dir1[i] for i, l in enumerate(label1) if l in use_list],
        [label1[i] for i, l in enumerate(label1) if l in use_list],
        cache=cache, resolution=resolution, cut=True)
    print('len train,test--',len(test_source))
    print('len train,test--',len(use_source))
    # print(train_source[0])
    print(test_source)
    print(use_source)
    return test_source,use_source