# -*- coding:utf-8 -*-
# author:peng
# Date：2023/4/4 12:01
import json
from glob import glob

import numpy as np
import torch
from PIL import Image
from facenet_pytorch import MTCNN

from facenet import Facenet

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# 人脸识别阈值
VERIFICATION_THRESHOLD = 0.8
# 检测人脸检测模型
mtcnn_detector = MTCNN(margin=20, keep_all=True, post_process=False, device=device)
# 加载人脸识别模型
face_model = Facenet()

def getFacenet():
    return face_model
# 识别人脸接口
def face_recognition(image_pil):
    # 加载已经注册的人脸
    save_pathes = glob('./face_save/*.jpg')

    faces = mtcnn_detector(image_pil)
    boxes, probs, landmarks = mtcnn_detector.detect(image_pil, landmarks=True)

    dis = []
    info_name = []
    if faces is not None:
        for face in faces:
            face_np = face.numpy().transpose(1, 2, 0)
            face = Image.fromarray(np.uint8(face_np))
            temp_dict = {}
            for save_path in save_pathes:
                save_img = Image.open(save_path).convert('RGB')
                l1 = face_model.detect_image(face, save_img)
                filename = save_path.split('\\')[-1]
                temp_dict[filename[:-4]] = l1
            dict = sorted(temp_dict.items(), key=lambda x: x[1])  # 按照l2从小到大排序
            if dict[0][1] < VERIFICATION_THRESHOLD:
                name = dict[0][0]
                dis.append(dict[0][1])
                info_name.append(name)
            else:
                dis.append(dict[0][1])
                info_name.append('unknown')
    return info_name, dis, boxes, landmarks



def list_to_json(ls):
    return json.dumps(ls)


if __name__ == '__main__':
    path = './img/peng2.jpg'
    img = Image.open(path).convert('RGB')
    info,dis,box,lan = face_recognition(img)
    print(info)
    print(dis)
    print(box)
    print(lan)