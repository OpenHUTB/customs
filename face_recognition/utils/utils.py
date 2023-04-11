import numpy as np
import math
import cv2

#-------------------------------------#
#   人脸对齐
#-------------------------------------#
def Alignment_1(imgs, landmarks):


    for img, landmark in zip(imgs, landmarks):

        img = np.array(img)
        x = landmark[0,0] - landmark[1,0]
        y = landmark[0,1] - landmark[1,1]

        if x==0:
            angle = 0
        else:
            angle = math.atan(y/x)*180/math.pi

        center = (img.shape[1]//2, img.shape[0]//2)

        RotationMatrix = cv2.getRotationMatrix2D(center, angle, 1)
        new_img = cv2.warpAffine(img,RotationMatrix,(img.shape[1],img.shape[0]))
        new_img = np.stack(new_img, axis=0)

    # RotationMatrix = np.array(RotationMatrix)
    # new_landmark = []
    # for i in range(landmark.shape[0]):
    #     pts = []
    #     pts.append(RotationMatrix[0,0]*landmark[i,0]+RotationMatrix[0,1]*landmark[i,1]+RotationMatrix[0,2])
    #     pts.append(RotationMatrix[1,0]*landmark[i,0]+RotationMatrix[1,1]*landmark[i,1]+RotationMatrix[1,2])
    #     new_landmark.append(pts)
    #
    # new_landmark = np.array(new_landmark)

    # return new_img, new_landmark
    return new_img