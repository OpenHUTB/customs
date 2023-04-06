# -*- coding:utf-8 -*-
# author:peng
# Date：2023/4/3 15:06
# Create face detector
import cv2
import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN
from matplotlib import pyplot as plt

mtcnn = MTCNN(keep_all=True, device='cpu')

# Load a single image and display
v_cap = cv2.VideoCapture('../img/aagfhgtpmv.mp4')
success, frame = v_cap.read()
frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
frame = Image.fromarray(frame)
mtcnn(frame, save_path='../face_save/{}.jpg'.format('Bob'))

# Detect face
boxes, probs, landmarks = mtcnn.detect(frame, landmarks=True)
print(boxes, probs, landmarks)
print(type(boxes), type(probs), type(landmarks))
print(probs.shape)

# Visualize
fig, ax = plt.subplots(figsize=(16, 12))
ax.imshow(frame)
ax.axis('off')

# for box, landmark in zip(boxes, landmarks):
#     ax.scatter(*np.meshgrid(box[[0, 2]], box[[1, 3]]))
#     ax.scatter(landmark[:, 0], landmark[:, 1], s=8, color='red')
# fig.show()
image_cv = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
for box in boxes:
    print(box)
    print(int(box[0]), int(box[1]),type(box[0]))

    cv2.rectangle(image_cv, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 0))
    # cv2.rectangle(image_cv,(336,165),(561,457),(255,0,0))

cv2.imshow('img',image_cv)
cv2.waitKey(0)
cv2.destroyAllWindows()
