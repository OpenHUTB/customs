# -*- coding:utf-8 -*-
# author:peng
# Date：2023/7/3 19:19
import math
import os

import cv2
import numpy as np
import torch
from PIL import Image, ImageFont, ImageDraw
from torchvision.ops import nms
from utils.config import cfg_mnet


# -----------------------------#
#   中心解码，宽高解码
# -----------------------------#
def decode(loc, priors, variances):
    boxes = torch.cat((priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
                       priors[:, 2:] * torch.exp(loc[:, 2:] * variances[1])), 1)
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes


# -----------------------------#
#   关键点解码
# -----------------------------#
def decode_landm(pre, priors, variances):
    landms = torch.cat((priors[:, :2] + pre[:, :2] * variances[0] * priors[:, 2:],
                        priors[:, :2] + pre[:, 2:4] * variances[0] * priors[:, 2:],
                        priors[:, :2] + pre[:, 4:6] * variances[0] * priors[:, 2:],
                        priors[:, :2] + pre[:, 6:8] * variances[0] * priors[:, 2:],
                        priors[:, :2] + pre[:, 8:10] * variances[0] * priors[:, 2:],
                        ), dim=1)
    return landms


# ------------------------------#
#    非极大值抑制
# ------------------------------#
def non_max_suppression(detection, conf_thres=0.5, nms_thres=0.3):
    # ------------------------------------------#
    #   找出该图片中得分大于门限函数的框。
    #   在进行重合框筛选前就
    #   进行得分的筛选可以大幅度减少框的数量。
    # ------------------------------------------#
    mask = detection[:, 4] >= conf_thres
    detection = detection[mask]

    if len(detection) <= 0:
        return []

    # ------------------------------------------#
    #   使用官方自带的非极大抑制会速度更快一些！
    # ------------------------------------------#
    keep = nms(
        detection[:, :4],
        detection[:, 4],
        nms_thres
    )
    best_box = detection[keep]
    return best_box.cpu().numpy()


# -----------------------------------#
#    retinaface网络的数据预处理
# -----------------------------------#
def preprocess_input(image):
    image -= np.array((104, 117, 123), np.float32)
    return image


# ---------------------------------------------------#
#   对输入图像进行resize
#   image: PIL.Image
# ---------------------------------------------------#
def resize_image(image, size, letterbox_image):
    iw, ih = image.size
    w, h = size
    if letterbox_image:
        scale = min(w / iw, h / ih)
        nw = int(iw * scale)
        nh = int(ih * scale)

        image = image.resize((nw, nh), Image.BICUBIC)
        new_image = Image.new('RGB', size, (128, 128, 128))
        new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))
    else:
        new_image = image.resize((w, h), Image.BICUBIC)
    return new_image


# ---------------------------------------------------#
#    人脸对齐
#    img: np.array
# ---------------------------------------------------#
def Alignment_1(img, landmark):
    global x, y
    if landmark.shape[0] == 68:
        x = landmark[36, 0] - landmark[45, 0]
        y = landmark[36, 1] - landmark[45, 1]
    elif landmark.shape[0] == 5:
        x = landmark[0, 0] - landmark[1, 0]
        y = landmark[0, 1] - landmark[1, 1]
    # 眼睛连线相对于水平线的倾斜角
    if x == 0:
        angle = 0
    else:
        # 计算它的弧度制
        angle = math.atan(y / x) * 180 / math.pi

    center = (img.shape[1] // 2, img.shape[0] // 2)

    RotationMatrix = cv2.getRotationMatrix2D(center, angle, 1)
    # 仿射函数
    new_img = cv2.warpAffine(img, RotationMatrix, (img.shape[1], img.shape[0]))

    RotationMatrix = np.array(RotationMatrix)
    new_landmark = []
    for i in range(landmark.shape[0]):
        pts = []
        pts.append(RotationMatrix[0, 0] * landmark[i, 0] + RotationMatrix[0, 1] * landmark[i, 1] + RotationMatrix[0, 2])
        pts.append(RotationMatrix[1, 0] * landmark[i, 0] + RotationMatrix[1, 1] * landmark[i, 1] + RotationMatrix[1, 2])
        new_landmark.append(pts)

    new_landmark = np.array(new_landmark)
    return new_img, new_landmark


# ---------------------------------------------------#
#   对输入图像进行resize
#   image: np.array
# ---------------------------------------------------#
def letterbox_image(image, size):
    ih, iw, _ = np.shape(image)
    w, h = size
    scale = min(w / iw, h / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)

    image = cv2.resize(image, (nw, nh))
    new_image = np.ones([size[1], size[0], 3]) * 128
    new_image[(h - nh) // 2:nh + (h - nh) // 2, (w - nw) // 2:nw + (w - nw) // 2] = image
    return new_image


# -----------------------------------------------------------------#
#   将输出调整为相对于原图的大小
#   result: n, 4+1+10
# -----------------------------------------------------------------#
def retinaface_correct_boxes(result, input_shape, image_shape):
    new_shape = image_shape * np.min(input_shape / image_shape)

    offset = (input_shape - new_shape) / 2. / input_shape
    scale = input_shape / new_shape

    scale_for_boxs = [scale[1], scale[0], scale[1], scale[0]]
    scale_for_landmarks = [scale[1], scale[0], scale[1], scale[0], scale[1], scale[0], scale[1], scale[0], scale[1],
                           scale[0]]

    offset_for_boxs = [offset[1], offset[0], offset[1], offset[0]]
    offset_for_landmarks = [offset[1], offset[0], offset[1], offset[0], offset[1], offset[0], offset[1], offset[0],
                            offset[1], offset[0]]

    result[:, :4] = (result[:, :4] - np.array(offset_for_boxs)) * np.array(scale_for_boxs)
    result[:, 5:] = (result[:, 5:] - np.array(offset_for_landmarks)) * np.array(scale_for_landmarks)
    return result


# ----------------------------------------#
#    在人脸在画框和写上置信度
#    old_image: np.array
# ----------------------------------------#
def put_box_img(old_image, boxes_conf_landms, index, known_face_names, face_encodings):
    # 和数据库中的特征图比较
    face_names = []

    distances, indexs = index.search(np.array(face_encodings), 1)
    name = "Unknown"
    for index, distance in zip(indexs, distances):
        if distance[0] < cfg_mnet["faiss distance"]:
            name = known_face_names[index[0]]
        face_names.append(name)

    for i, b in enumerate(boxes_conf_landms):
        text = "confidence:{:.4f}".format(b[4]) + '   name:{}'.format(face_names[i])
        print('text:', text)
        b = list(map(int, b))
        # ---------------------------------------------------#
        #   b[0]-b[3]为人脸框的坐标，b[4]为得分
        # ---------------------------------------------------#
        cv2.rectangle(old_image, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 3)
        cx = b[0]
        cy = b[1] + 12
        # 只能写英文
        # cv2.putText(old_image, text, (cx, cy),
        #             cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 0, 0))
        old_image = cv2ImgAddText(old_image, text, cx, cy)

        # ---------------------------------------------------#
        #   b[5]-b[14]为人脸关键点的坐标
        # ---------------------------------------------------#
        cv2.circle(old_image, (b[5], b[6]), 1, (0, 0, 255), 4)
        cv2.circle(old_image, (b[7], b[8]), 1, (0, 255, 255), 4)
        cv2.circle(old_image, (b[9], b[10]), 1, (255, 0, 255), 4)
        cv2.circle(old_image, (b[11], b[12]), 1, (0, 255, 0), 4)
        cv2.circle(old_image, (b[13], b[14]), 1, (255, 0, 0), 4)
    return old_image


# --------------------------------------#
#   写中文需要转成PIL来写。
# --------------------------------------#
def cv2ImgAddText(img, label, left, top, textColor=(255, 0, 0)):
    img = Image.fromarray(np.uint8(img))
    # ---------------#
    #   设置字体
    # ---------------#
    font = ImageFont.truetype(font='model_data/simhei.ttf', size=20)

    draw = ImageDraw.Draw(img)
    label = label.encode('utf-8')
    draw.text((left, top), str(label, 'UTF-8'), fill=textColor, font=font)
    return np.asarray(img)


# ---------------------------------#
#   比较人脸
#   known_face_encodings: n, 512
# ---------------------------------#
def compare_faces(known_face_encodings, face_encoding_to_check, tolerance=cfg_mnet['facenet_threhold']):
    if len(known_face_encodings) == 0:
        return np.empty((0))
    dis = np.linalg.norm(known_face_encodings - face_encoding_to_check, axis=1)
    return list(dis <= tolerance), dis
