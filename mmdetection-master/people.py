from torch import device
import torch
import os
from PIL import Image
import numpy as np
from mmdet.apis import init_detector, show_result_pyplot, inference_detector
def fnsplit(filename):
    for i in range(len(filename)):
        filename[i] = filename[i].split('.')[0]
    return filename


config_file = 'configs/mask_rcnn/mask_rcnn_r50_fpn_mstrain-poly_3x_coco.py'

checkpoint_file = 'checkpoints/mask_rcnn_r50_fpn_mstrain-poly_3x_coco_20210524_201154-21b550bb.pth'

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model = init_detector(config_file, checkpoint_file, device=device)
path = "视频帧文件路径"
pathname = os.listdir(path)
for name in pathname:
    filepath = path + '/' + name
    filename = os.listdir(filepath)
    filename1 = fnsplit(filename)
    filename1 = list(map(int, filename1))
    filename1.sort()

    savepath = "步态轮廓图文件存储路径" + '/' +name

    if not os.path.isdir(savepath):
        os.makedirs(savepath)
    for fn in filename1:
        img_file = filepath + "/" + str(fn) + '.png'
        print(img_file)
        result = inference_detector(model, img_file)
        image = Image.open(img_file)
        # 将图片转换为numpy数组
        np_img = np.array(image)
        a, b, c = np_img.shape
        # print(a,b,c)
        for i in range(0, a):
            for j in range(0, b):
                if result[1][0][0][i, j] == True:
                    np_img[i, j] = [255, 255, 255]
                else:
                    np_img[i, j] = [0, 0, 0]
        np_img = np_img.astype(np.uint8)
        img_kou = Image.fromarray(np_img)
        # img_kou = img_kou.resize((64, 64), Image.ANTIALIAS)
        img_kou.save(savepath + "/" + str(fn) + ".png")
    print("complete")