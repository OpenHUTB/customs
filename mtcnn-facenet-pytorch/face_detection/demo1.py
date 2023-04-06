# -*- coding:utf-8 -*-
# author:peng
# Date：2023/3/31 20:51
import cv2

# MTCNN和MobileFaceNet
# 检测人脸
def face_detection(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 加载OpenCV人脸检测分类器Haar
    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    face = face_cascade.detectMultiScale(gray_img,1.3,5)
    for x, y, w, h in face:
        cv2.rectangle(img, (x, y), (x + w, y + h), color=(0, 0, 255))
        cv2.imwrite("./data/User." + str(1) + '.' + str(1) + '.jpg', img[y:y + h, x:x + w])
    cv2.imshow('img', img)


# img = cv2.imread('../img/FudanPed00042.png')
# face_detection(img)

cap = cv2.VideoCapture(0)

while True:
    flag,frame = cap.read()

    if not flag:
        break
    face_detection(frame)

    if ord(' ') == cv2.waitKey(0):
        break


cap.release()

cv2.destroyAllWindows()
