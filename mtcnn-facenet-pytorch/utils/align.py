# -*- coding:utf-8 -*-
# author:peng
# Date：2023/4/5 21:28
import math

import cv2
import numpy as np


# -------------------------------------#
#   人脸对齐
# -------------------------------------#
def Alignment_1(img, landmark, box):
    x,y = 0,0
    if landmark.shape[0] == 68:
        x = landmark[36, 0] - landmark[45, 0]
        y = landmark[36, 1] - landmark[45, 1]
    elif landmark.shape[0] == 5:
        x = landmark[0, 0] - landmark[1, 0]
        y = landmark[0, 1] - landmark[1, 1]

    if x == 0:
        angle = 0
    else:
        angle = math.atan(y / x) * 180 / math.pi

    center = (img.shape[1] // 2, img.shape[0] // 2)

    RotationMatrix = cv2.getRotationMatrix2D(center, angle, 1)
    new_img = cv2.warpAffine(img, RotationMatrix, (img.shape[1], img.shape[0]))

    RotationMatrix = np.array(RotationMatrix)
    new_landmark = []
    for i in range(landmark.shape[0]):
        pts = []
        pts.append(RotationMatrix[0, 0] * landmark[i, 0] + RotationMatrix[0, 1] * landmark[i, 1] + RotationMatrix[0, 2])
        pts.append(RotationMatrix[1, 0] * landmark[i, 0] + RotationMatrix[1, 1] * landmark[i, 1] + RotationMatrix[1, 2])
        new_landmark.append(pts)
    new_landmark = np.array(new_landmark)

    new_box = []
    new_box.append(RotationMatrix[0, 0] * box[0] + RotationMatrix[0, 1] * box[1] + RotationMatrix[0, 2])
    new_box.append(RotationMatrix[1, 0] * box[0] + RotationMatrix[1, 1] * box[1] + RotationMatrix[1, 2])

    new_box.append(RotationMatrix[0, 0] * box[0+2] + RotationMatrix[0, 1] * box[1+2] + RotationMatrix[0, 2])
    new_box.append(RotationMatrix[1, 0] * box[0+2] + RotationMatrix[1, 1] * box[1+2] + RotationMatrix[1, 2])
    new_box = np.array(new_box)

    return new_img, new_landmark, new_box


def Alignment_2(img, std_landmark, landmark):
    def Transformation(std_landmark, landmark):
        std_landmark = np.matrix(std_landmark).astype(np.float64)
        landmark = np.matrix(landmark).astype(np.float64)

        c1 = np.mean(std_landmark, axis=0)
        c2 = np.mean(landmark, axis=0)
        std_landmark -= c1
        landmark -= c2

        s1 = np.std(std_landmark)
        s2 = np.std(landmark)
        std_landmark /= s1
        landmark /= s2

        U, S, Vt = np.linalg.svd(std_landmark.T * landmark)
        R = (U * Vt).T

        return np.vstack([np.hstack(((s2 / s1) * R, c2.T - (s2 / s1) * R * c1.T)), np.matrix([0., 0., 1.])])

    Trans_Matrix = Transformation(std_landmark, landmark)  # Shape: 3 * 3
    Trans_Matrix = Trans_Matrix[:2]
    Trans_Matrix = cv2.invertAffineTransform(Trans_Matrix)
    new_img = cv2.warpAffine(img, Trans_Matrix, (img.shape[1], img.shape[0]))

    Trans_Matrix = np.array(Trans_Matrix)
    new_landmark = []
    for i in range(landmark.shape[0]):
        pts = []
        pts.append(Trans_Matrix[0, 0] * landmark[i, 0] + Trans_Matrix[0, 1] * landmark[i, 1] + Trans_Matrix[0, 2])
        pts.append(Trans_Matrix[1, 0] * landmark[i, 0] + Trans_Matrix[1, 1] * landmark[i, 1] + Trans_Matrix[1, 2])
        new_landmark.append(pts)

    new_landmark = np.array(new_landmark)

    return new_img, new_landmark
