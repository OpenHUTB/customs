# -*- coding:utf-8 -*-
# author:peng
# Date：2023/4/3 14:32
from facenet_pytorch import MTCNN
import cv2
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from tqdm.notebook import tqdm

# Create face detector
mtcnn = MTCNN(margin=20, keep_all=True, post_process=False, device='cpu')

img = Image.open('../img/FudanPed00042.png').convert('RGB')
faces = mtcnn(img)

# Visualize
if len(faces) > 1:
    fig, axes = plt.subplots(1, len(faces))
    for face, ax in zip(faces, axes):
        ax.imshow(face.permute(1, 2, 0).int().numpy())
        ax.axis('off')
    fig.show()
else:
    fig, axes = plt.subplots(1, len(faces))
    axes.imshow(faces[0].int().numpy().transpose(1, 2, 0))
    axes.axis('off')
    fig.show()