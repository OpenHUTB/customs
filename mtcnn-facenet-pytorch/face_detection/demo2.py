# -*- coding:utf-8 -*-
# author:peng
# Date：2023/4/3 10:52
import cv2
from facenet_pytorch import MTCNN
from matplotlib import pyplot as plt

# Create face detector
from utils import align

mtcnn = MTCNN(margin=20, keep_all=True, post_process=False, device='cpu')


v_cap = cv2.VideoCapture('../img/aagfhgtpmv.mp4')
success, frame = v_cap.read()
frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

boxes,pro,landmarks = mtcnn.detect(frame,landmarks=True)
print(boxes)
print(pro)
print(landmarks,type(landmarks))
new_frame,new_landmarks = align.Alignment_1(frame,landmarks)

plt.figure(figsize=(12, 8))
for mark in new_landmarks:
    for item in mark:
        x, y = item
        cv2.circle(new_frame, (int(x), int(y)), 1, (0, 255, 0), 4)
plt.imshow(new_frame)
plt.axis('off')
plt.show()

# plt.figure(figsize=(12, 8))
# for box in boxes:
#     x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
#     img_cut = frame[x1:x2, y1:y2]
# plt.imshow(img_cut)
# plt.axis('off')
# plt.show()
#
# # Detect face
# faces = mtcnn(frame)
# print(type(faces), type(faces[0]), faces.shape) # <class 'torch.Tensor'> <class 'torch.Tensor'> torch.Size([1, 3, 160, 160])
# # Visualize
# if len(faces) > 1:
#     fig, axes = plt.subplots(1, len(faces))
#     for face, ax in zip(faces, axes):
#         ax.imshow(face.permute(1, 2, 0).int().numpy())
#         ax.axis('off')
#     fig.show()
# else:
#     fig, axes = plt.subplots(1, len(faces))
#     axes.imshow(faces[0].int().numpy().transpose(1, 2, 0))
#     axes.axis('off')
#     fig.show()
