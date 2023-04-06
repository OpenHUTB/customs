# -*- coding:utf-8 -*-
# author:peng
# Date：2023/4/6 10:45
# -*- coding:utf-8 -*-
# author:peng
# Date：2023/4/3 10:52
import cv2
from facenet_pytorch import MTCNN
from matplotlib import pyplot as plt

# Create face detector
from utils import align

mtcnn = MTCNN(margin=20, keep_all=True, post_process=False, device='cpu')


# v_cap = cv2.VideoCapture('./img/aagfhgtpmv.mp4')
# success, frame = v_cap.read()
# frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
frame = cv2.imread('img/peng3.jpg')
frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

boxes,pro,landmarks = mtcnn.detect(frame,landmarks=True)
print(boxes)
print(pro)
print(landmarks,type(landmarks))
new_frame,new_landmarks,new_box = align.Alignment_1(frame,landmarks[0], boxes[0])
print(new_landmarks,type(new_landmarks))
print(new_box,type(new_box))

plt.figure(figsize=(12, 8))
plt.subplot(121)
for mark in new_landmarks:
    x, y = mark
    cv2.circle(new_frame, (int(x), int(y)), 2, (0, 255, 0), 4)

x1, y1, x2, y2 = int(new_box[0]), int(new_box[1]), int(new_box[2]), int(new_box[3])
cv2.rectangle(new_frame, (x1, y1), (x2, y2), (0, 0, 255), 6)
plt.imshow(new_frame)

plt.axis('off')
plt.subplot(122)
for mark in landmarks:
    for item in mark:
        x, y = item
        cv2.circle(frame, (int(x), int(y)), 2, (255, 0, 255), 4)
x1, y1, x2, y2 = int(boxes[0][0]), int(boxes[0][1]), int(boxes[0][2]), int(boxes[0][3])
cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 6)
plt.imshow(frame)
plt.axis('off')
plt.show()
