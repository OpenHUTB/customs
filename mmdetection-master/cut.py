#-*-coding:gb2312-*-
import os
from PIL import Image
import numpy as np


def cut_image(path, cut_path, size):
    '''
    剪切图片
    :param path: 输入图片路径
    :param cut_path: 剪切图片后的输出路径
    :param size: 要剪切的图片大小
    :return:
    '''
    for (root, dirs, files) in os.walk(path):
        temp = root.replace(path, cut_path)
        if not os.path.exists(temp):
            os.makedirs(temp)
        for file in files:
            image, flag = cut(Image.open(os.path.join(root, file)))
            if not flag: Image.fromarray(image).convert('L').resize((size, size)).save(os.path.join(temp, file))
        print(temp)
    pass


def cut(image):
    '''
    通过找到人的最小最大高度与宽度把人的轮廓分割出来，、
    因为原始轮廓图为二值图，因此头顶为将二值图像列相加后，形成一列后第一个像素值不为0的索引。
    同理脚底为形成一列后最后一个像素值不为0的索引。
    人的宽度也同理。
    :param image: 需要裁剪的图片 N*M的矩阵
    :return: temp:裁剪后的图片 size*size的矩阵。flag：是否是符合要求的图片
    '''
    image = image.convert('L')
    print(image.size)
    image = np.array(image)
    # 找到人的最小最大高度与宽度
    height_min = (image.sum(axis=1) != 0).argmax()
    height_max = ((image.sum(axis=1) != 0).cumsum()).argmax()
    width_min = (image.sum(axis=0) != 0).argmax()
    width_max = ((image.sum(axis=0) != 0).cumsum()).argmax()
    head_top = image[height_min, :].argmax()
    # 设置切割后图片的大小，为size*size，因为人的高一般都会大于宽
    size = height_max - height_min
    temp = np.zeros((size, size))
    # 将width_max-width_min（宽）乘height_max-height_min（高，szie）的人的轮廓图，放在size*size的图片中央
    # l = (width_max-width_min)//2
    # r = width_max-width_min-l
    # 以头为中心，将将width_max-width_min（宽）乘height_max-height_min（高，szie）的人的轮廓图，放在size*size的图片中央
    l1 = head_top - width_min
    r1 = width_max - head_top
    # 若宽大于高，或头的左侧或右侧身子比要生成图片的一般要大。则此图片为不符合要求的图片
    flag = False
    if size <= width_max - width_min or size // 2 < r1 or size // 2 < l1:
        flag = True
        return temp, flag
    temp[:, (size // 2 - l1):(size // 2 + r1)] = image[height_min:height_max, width_min:width_max]
    return temp, flag



if __name__ == '__main__':
    path = '步态轮廓图存储路径'
    path1 = os.listdir(path)
    for name in path1:
        input1 = path + '/' + name
        print(input1)
        output = input1
        cut_image(input1, output, 64)
        path_list = os.listdir(output)
        path_list.sort(key=lambda x: int(x.split('.')[0]))
        i = 0
        for name1 in path_list:
            if i >= 100:
                os.remove(output + '/' + name1)
            i+=1
    print('success')