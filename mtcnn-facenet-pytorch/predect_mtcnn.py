# -*- coding:utf-8 -*-
# author:peng
# Date：2023/4/3 14:45

from glob import glob

import cv2
import numpy as np
import torch.cuda
from PIL import Image
from facenet_pytorch import MTCNN
from matplotlib import pyplot as plt

from facenet import Facenet

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def face_register(image, name):
    # image = Image.open(img_path).convert('RGB')
    mtcnn_detector = MTCNN(margin=20, keep_all=True, post_process=False, device=device)
    boxes, probs, landmarks = mtcnn_detector.detect(image, landmarks=True)
    if probs.shape[0] is not 0:
        # Visualize
        fig, ax = plt.subplots(figsize=(16, 12))
        ax.imshow(image)
        ax.axis('off')

        for box, landmark in zip(boxes, landmarks):
            ax.scatter(*np.meshgrid(box[[0, 2]], box[[1, 3]]), color='green')
            ax.scatter(landmark[:, 0], landmark[:, 1], s=8, color='red')
        fig.show()

        if probs.shape[0] == 1:
            mtcnn_detector(image, save_path='./face_save/{}.jpg'.format(name))
            print("注册成功！")
        else:
            print('注册图片有错，图片中有且只有一个人脸')
    else:
        print('注册图片有错，图片中有且只有一个人脸')


def face_register_capture(name):
    print("点击空格确认拍照！")
    mtcnn_detector = MTCNN(margin=20, keep_all=True, post_process=False, device=device)

    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        cv2.imshow('image', frame)
        if cv2.waitKey(1) & 0xFF == ord(' '):
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            boxes, probs, landmarks = mtcnn_detector.detect(image, landmarks=True)
            if probs.shape[0] is not 0:
                # Visualize
                fig, ax = plt.subplots(figsize=(16, 12))
                ax.imshow(image)
                ax.axis('off')

                for box, landmark in zip(boxes, landmarks):
                    ax.scatter(*np.meshgrid(box[[0, 2]], box[[1, 3]]), s=60, color='green')
                    ax.scatter(landmark[:, 0], landmark[:, 1], s=18, color='red')
                fig.show()

                if probs.shape[0] == 1:
                    mtcnn_detector(image, save_path='./face_save/{}.jpg'.format(name))
                    print("注册成功！")
                else:
                    print('注册图片有错，图片中有且只有一个人脸')
            else:
                print('注册图片有错，图片中有且只有一个人脸')
            break
    cap.release()
    cv2.destroyAllWindows()


"""
人脸识别是通过图像路径读取将要识别的人脸，通过经过MTCNN的检测人脸和对其，
在使用MobileFaceNet预测人脸的特征，最终得到特征和人脸库中的特征值比较相似度，最终得到阈值超过0.8的最高相似度结果，
对应的名称就是该人脸识别的结果。最后把结果在图像中画框和标记上名称并显示出来。
"""
def face_recognition(image_pil):
    global image_cv
    VERIFICATION_THRESHOLD = 0.8

    mtcnn_detector = MTCNN(margin=20, keep_all=True, post_process=False, device=device)
    face_model = Facenet()

    faces = mtcnn_detector(image_pil)
    boxes, probs, landmarks = mtcnn_detector.detect(image_pil, landmarks=True)

    save_pathes = glob('./face_save/*.jpg')
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

    for k, box in enumerate(boxes):
        # font = ImageFont.truetype('font/simfang.ttf', 18, encoding="utf-8")
        distance = '%.2f' % dis[k]
        label = "{}, {}".format(info_name[k], distance)
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])

        image_cv = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
        cv2.putText(image_cv, label, (x1, y1 - 15), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255, 0, 0), 4, cv2.LINE_AA)
        cv2.rectangle(image_cv, (x1, y1), (x2, y2), (0, 0, 255), 6)
    image_pil = Image.fromarray(cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB))

    # Visualize
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.imshow(image_pil)
    plt.axis('off')
    info_name = list(filter(lambda x: x != 'unknown', info_name))
    if len(info_name) != 0:
        for i in range(len(info_name)):
            name = info_name[i]
            image = Image.open('./face_save/{}.jpg'.format(name)).convert('RGB')
            plt.subplot(2, len(info_name), len(info_name) + i + 1)
            plt.imshow(image)
            plt.axis('off')
    plt.show()


def face_recognition_capture():
    global image_cv
    VERIFICATION_THRESHOLD = 0.8

    print("点击空格确认拍照！")
    mtcnn_detector = MTCNN(margin=20, keep_all=True, post_process=False, device=device)
    face_model = Facenet()

    save_pathes = glob('./face_save/*.jpg')
    dis = []
    info_name = []

    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        cv2.imshow('image', frame)
        if cv2.waitKey(1) & 0xFF == ord(' '):
            image_pil = Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
            faces = mtcnn_detector(image_pil)
            boxes, probs, landmarks = mtcnn_detector.detect(image_pil, landmarks=True)
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

            for k, box in enumerate(boxes):
                # font = ImageFont.truetype('font/simfang.ttf', 18, encoding="utf-8")
                distance = '%.2f' % dis[k]
                label = "{}, {}".format(info_name[k], distance)
                x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])

                image_cv = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
                cv2.putText(image_cv, label, (x1, y1 - 15), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255, 0, 0), 4, cv2.LINE_AA)
                cv2.rectangle(image_cv, (x1, y1), (x2, y2), (0, 0, 255), 6)

            image_pil = Image.fromarray(cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB))

            # Visualize
            plt.figure()
            plt.subplot(2, 1, 1)
            plt.imshow(image_pil)
            plt.axis('off')

            info_name = list(filter(lambda x: x != 'unknown', info_name))
            if len(info_name) != 0:
                for i in range(len(info_name)):
                    name = info_name[i]
                    image = Image.open('./face_save/{}.jpg'.format(name)).convert('RGB')
                    plt.subplot(2, len(info_name), len(info_name) + i + 1)
                    plt.imshow(image)
                    plt.axis('off')
            plt.show()
            break
    cap.release()
    cv2.destroyAllWindows()



if __name__ == '__main__':
    # i = int(input("请选择功能，1为注册人脸，2为识别人脸："))
    # image_path = input("请输入图片路径：")
    # image = Image.open(image_path).convert('RGB')
    # if i == 1:
    #     user_name = input("请输入注册名：")
    #     face_register(image, user_name)
    # elif i == 2:
    #     face_recognition(image)
    # else:
    #     print("功能选择错误")
    # face_register_capture('peng_camera')
    face_recognition_capture()
