# -*- coding:utf-8 -*-
# author:peng
# Date：2023/4/3 19:36
import cv2
import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN
from matplotlib import pyplot as plt
from torchvision import transforms

image_path = '../img/peng2.jpg'
mtcnn = MTCNN(margin=20, keep_all=True, post_process=False, device='cpu')
img = Image.open(image_path).convert('RGB')
img_tensor = mtcnn(img)
img_pil = transforms.ToPILImage()(img_tensor[0])
img_pil.save('peng_trans.jpg')

img_np = img_tensor[0].numpy().transpose(1,2,0)
img_pil2 = Image.fromarray(np.uint8(img_np))

plt.figure()
plt.subplot(1,3,1)
plt.imshow(img_pil)
plt.subplot(1,3,2)
plt.imshow(img_tensor[0].permute(1, 2, 0).int().numpy())
plt.axis('off')
plt.subplot(1,3,3)
plt.imshow(img_pil2)
plt.axis('off')
plt.show()
