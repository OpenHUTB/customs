# -*- coding:utf-8 -*-
# author:peng
# Date：2023/7/3 20:23

import os

import numpy as np

from retinaface_arcface import retinaface_arcface


# ------------------------------------#
#  将datasets文件夹中图片转为.npy
# ------------------------------------#
def encoding():
    retinaface = retinaface_arcface()
    list_dir = os.listdir("datasets")
    image_paths = []
    names = []
    for name in list_dir:
        image_paths.append("datasets/" + name)
        names.append(name.split(".")[0])
        print(name.split(".")[0])
    retinaface.encode_face_dataset(image_paths, names)


# ------------------------------------#
#  将图片特征向量添加进npy文件
#  image: np.array RGB
#  name : str
# ------------------------------------#
def encoding_one(image_ls, name_ls):
    if not os.path.exists('model_data/face_encoding.npy') or not os.path.exists('model_data/face_name.npy'):
        print('先加载数据库中图片')
        return
    old_face_encodings = np.load('model_data/face_encoding.npy')
    old_name_encodings = np.load('model_data/face_name.npy')

    retinaface = retinaface_arcface()
    new_face_encodings = []
    new_face_names = []
    for face, name in zip(image_ls, name_ls):
        boxes_conf_landms, face_encodings = retinaface.detect_image(face)
        if boxes_conf_landms is not None:
            print(boxes_conf_landms.shape, len(face_encodings))
        else:
            print(name, '未检测到人脸')
            continue
        # ---------------------------------------------------#
        #   选取最大的人脸框。
        # ---------------------------------------------------#
        best_face_index = None
        biggest_area = 0
        for i, result in enumerate(boxes_conf_landms):
            left, top, right, bottom = result[0:4]

            w = right - left
            h = bottom - top
            if w * h > biggest_area:
                biggest_area = w * h
                best_face_index = i
        face_encoding = face_encodings[best_face_index]
        new_face_encodings.append(face_encoding)
        new_face_names.append(name)
    new_face_encodings = np.array(new_face_encodings)
    new_face_names = np.array(new_face_names)

    update_face = np.concatenate((old_face_encodings, new_face_encodings))
    update_name = np.concatenate((old_name_encodings, new_face_names))
    np.save('model_data/face_encoding.npy', update_face)
    np.save('model_data/face_name.npy', update_name)


if __name__ == '__main__':
    encoding()
