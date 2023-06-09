import torch.utils.data as tordata
import numpy as np
import os.path as osp
import os
import pickle
import cv2
import xarray as xr

# class DataSet(tordata.Dataset):
#     def __init__(self, seq_dir, label, seq_type, view, cache, resolution , cut=False):
#         self.seq_dir = seq_dir
#         self.view = view
#         self.seq_type = seq_type
#         self.label = label
#         self.cache = cache
#         self.resolution = int(resolution)
#         self.cut_padding = int(float(resolution)/64*10)
#         self.data_size = len(self.label)
#         self.data = [None] * self.data_size
#         self.frame_set = [None] * self.data_size
#         self.cut =cut
#         self.label_set = set(self.label)
#         self.seq_type_set = set(self.seq_type)
#         self.view_set = set(self.view)
#         _ = np.zeros((len(self.label_set),
#                       len(self.seq_type_set),
#                       len(self.view_set))).astype('int')
#         _ -= 1
#
#         self.index_dict = xr.DataArray(
#             _,
#             coords={'label': sorted(list(self.label_set)),
#                     'seq_type': sorted(list(self.seq_type_set)),
#                     'view': sorted(list(self.view_set))},
#             dims=['label', 'seq_type', 'view'])
#         # print(self.index_dict.shape)
#         for i in range(self.data_size):
#             _label = self.label[i]
#             _seq_type = self.seq_type[i]
#             _view = self.view[i]
#             self.index_dict.loc[_label, _seq_type, _view] = i
#         # print(self.index_dict)
#     def load_all_data(self):
#         # print(self.cache)
#         for i in range(self.data_size):
#             if i % 10000 ==0:
#                 print('number-',i)
#             self.load_data(i)
#
#     def load_data(self, index):
#         return self.__getitem__(index)
#
#     def __loader__(self, path):
#         if self.cut:
#             return self.img2xarray(
#                 path)[:, :, self.cut_padding:-self.cut_padding].astype(
#                 'float32') / 255.0
#
#         else:
#             a = self.img2xarray(
#                 path).astype('float32') / 255.0
#             return a
#     def __getitem__(self, index):
#         # pose sequence sampling
#         # print(self.cache)
#         if not self.cache:
#             # print('-1-')
#             # print(self.seq_dir[index])
#             data = [self.__loader__(_path) for _path in self.seq_dir[index]]
#             frame_set = [set(feature.coords['frame'].values.tolist()) for feature in data]
#             frame_set = list(set.intersection(*frame_set))
#         elif self.data[index] is None:
#             # print('-2-')
#             data = [self.__loader__(_path) for _path in self.seq_dir[index]]
#             frame_set = [set(feature.coords['frame'].values.tolist()) for feature in data]
#             frame_set = list(set.intersection(*frame_set))
#             self.data[index] = data
#             self.frame_set[index] = frame_set
#         else:
#             # print('-3-')
#             data = self.data[index]
#             frame_set = self.frame_set[index]
#         # print(self.label[index], self.seq_type[index],self.view[
#         #     index],len(frame_set),data[0].shape)
#         return data, frame_set, self.view[
#             index], self.seq_type[index], self.label[index],
#
#     def img2xarray(self, flie_path):
#         imgs = sorted(list(os.listdir(flie_path)))
#
#         frame_list = [np.reshape(
#             cv2.imread(osp.join(flie_path, _img_path)),
#             [self.resolution, self.resolution, -1])[:, :, 0]
#                       for _img_path in imgs
#                       if osp.isfile(osp.join(flie_path, _img_path))]
#         num_list = list(range(len(frame_list)))
#         data_dict = xr.DataArray(
#             frame_list,
#             coords={'frame': num_list},
#             dims=['frame', 'img_y', 'img_x'],
#         )
#         return data_dict
#
#     def __len__(self):
#         return len(self.label)



class DataSet(tordata.Dataset):
    def __init__(self, seq_dir, label, view, cache, resolution , cut=False):
        self.seq_dir = seq_dir
        self.view = view
        self.label = label
        self.cache = cache
        self.resolution = int(resolution)
        self.cut_padding = int(float(resolution)/64*10)
        self.data_size = len(self.label)
        self.data = [None] * self.data_size
        self.frame_set = [None] * self.data_size
        self.cut =cut
        self.label_set = set(self.label)
        # self.seq_type_set = set(self.seq_type)
        self.view_set = set(self.view)
        _ = np.zeros((len(self.label_set),
                      len(self.view_set))).astype('int')
        _ -= 1

        self.index_dict = xr.DataArray(
            _,
            coords={'label': sorted(list(self.label_set)),
                    'view': sorted(list(self.view_set))},
            dims=['label', 'view'])
        # print(self.index_dict.shape)
        for i in range(self.data_size):
            _label = self.label[i]
            # _seq_type = self.seq_type[i]
            _view = self.view[i]
            self.index_dict.loc[_label, _view] = i
        # print(self.index_dict)
    def load_all_data(self):
        # print(self.cache)
        for i in range(self.data_size):
            if i % 10000 ==0:
                print('number-',i)
            self.load_data(i)

    def load_data(self, index):
        return self.__getitem__(index)

    def __loader__(self, path):
        if self.cut:
            return self.img2xarray(
                path)[:, :, self.cut_padding:-self.cut_padding].astype(
                'float32') / 255.0

        else:
            a = self.img2xarray(
                path).astype('float32') / 255.0
            return a
    def __getitem__(self, index):
        # pose sequence sampling
        # print(self.cache)
        if not self.cache:
            # print('-1-')
            # print(self.seq_dir[index])
            data = [self.__loader__(_path) for _path in self.seq_dir[index]]
            frame_set = [set(feature.coords['frame'].values.tolist()) for feature in data]
            frame_set = list(set.intersection(*frame_set))
        elif self.data[index] is None:
            # print('-2-')
            data = [self.__loader__(_path) for _path in self.seq_dir[index]]
            frame_set = [set(feature.coords['frame'].values.tolist()) for feature in data]
            frame_set = list(set.intersection(*frame_set))
            self.data[index] = data
            self.frame_set[index] = frame_set
        else:
            # print('-3-')
            data = self.data[index]
            frame_set = self.frame_set[index]
        # print(self.label[index], self.seq_type[index],self.view[
        #     index],len(frame_set),data[0].shape)
        return data, frame_set, self.view[
            index], self.label[index],

    def img2xarray(self, flie_path):
        imgs = sorted(list(os.listdir(flie_path)))

        frame_list = [np.reshape(
            cv2.imread(osp.join(flie_path, _img_path)),
            [self.resolution, self.resolution, -1])[:, :, 0]
                      for _img_path in imgs
                      if osp.isfile(osp.join(flie_path, _img_path))]
        num_list = list(range(len(frame_list)))
        data_dict = xr.DataArray(
            frame_list,
            coords={'frame': num_list},
            dims=['frame', 'img_y', 'img_x'],
        )
        return data_dict

    def __len__(self):
        return len(self.label)

class DataSet1(tordata.Dataset):
    def __init__(self, seq_dir, label, cache, resolution , cut=False):
        self.seq_dir = seq_dir
        self.label = label
        self.cache = cache
        self.resolution = int(resolution)
        self.cut_padding = int(float(resolution)/64*10)
        self.data_size = len(self.label)
        self.data = [None] * self.data_size
        self.frame_set = [None] * self.data_size
        self.cut =cut
        self.label_set = set(self.label)
        # self.seq_type_set = set(self.seq_type)
        _ = np.zeros((len(self.label_set))).astype('int')
        _ -= 1

        self.index_dict = xr.DataArray(
            _,
            coords={'label': sorted(list(self.label_set))},
            dims=['label'])
        # print(self.index_dict.shape)
        for i in range(self.data_size):
            _label = self.label[i]
            # _seq_type = self.seq_type[i]
            self.index_dict.loc[_label] = i
        # print(self.index_dict)
    def load_all_data(self):
        # print(self.cache)
        for i in range(self.data_size):
            if i % 10000 ==0:
                print('number-',i)
            self.load_data(i)

    def load_data(self, index):
        return self.__getitem__(index)

    def __loader__(self, path):
        if self.cut:
            return self.img2xarray(
                path)[:, :, self.cut_padding:-self.cut_padding].astype(
                'float32') / 255.0

        else:
            a = self.img2xarray(
                path).astype('float32') / 255.0
            return a
    def __getitem__(self, index):
        # pose sequence sampling
        # print(self.cache)
        if not self.cache:
            # print('-1-')
            # print(self.seq_dir[index])
            data = [self.__loader__(_path) for _path in self.seq_dir[index]]
            frame_set = [set(feature.coords['frame'].values.tolist()) for feature in data]
            frame_set = list(set.intersection(*frame_set))
        elif self.data[index] is None:
            # print('-2-')
            data = [self.__loader__(_path) for _path in self.seq_dir[index]]
            frame_set = [set(feature.coords['frame'].values.tolist()) for feature in data]
            frame_set = list(set.intersection(*frame_set))
            self.data[index] = data
            self.frame_set[index] = frame_set
        else:
            # print('-3-')
            data = self.data[index]
            frame_set = self.frame_set[index]
        # print(self.label[index], self.seq_type[index],self.view[
        #     index],len(frame_set),data[0].shape)
        return data, frame_set, self.label[index],

    def img2xarray(self, flie_path):
        imgs = sorted(list(os.listdir(flie_path)))

        frame_list = [np.reshape(
            cv2.imread(osp.join(flie_path, _img_path)),
            [self.resolution, self.resolution, -1])[:, :, 0]
                      for _img_path in imgs
                      if osp.isfile(osp.join(flie_path, _img_path))]
        num_list = list(range(len(frame_list)))
        data_dict = xr.DataArray(
            frame_list,
            coords={'frame': num_list},
            dims=['frame', 'img_y', 'img_x'],
        )
        return data_dict

    def __len__(self):
        return len(self.label)