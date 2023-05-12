#-*-coding:gb2312-*-
import os
from PIL import Image
import numpy as np


def cut_image(path, cut_path, size):
    '''
    ����ͼƬ
    :param path: ����ͼƬ·��
    :param cut_path: ����ͼƬ������·��
    :param size: Ҫ���е�ͼƬ��С
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
    ͨ���ҵ��˵���С���߶����Ȱ��˵������ָ��������
    ��Ϊԭʼ����ͼΪ��ֵͼ�����ͷ��Ϊ����ֵͼ������Ӻ��γ�һ�к��һ������ֵ��Ϊ0��������
    ͬ��ŵ�Ϊ�γ�һ�к����һ������ֵ��Ϊ0��������
    �˵Ŀ��Ҳͬ��
    :param image: ��Ҫ�ü���ͼƬ N*M�ľ���
    :return: temp:�ü����ͼƬ size*size�ľ���flag���Ƿ��Ƿ���Ҫ���ͼƬ
    '''
    image = image.convert('L')
    print(image.size)
    image = np.array(image)
    # �ҵ��˵���С���߶�����
    height_min = (image.sum(axis=1) != 0).argmax()
    height_max = ((image.sum(axis=1) != 0).cumsum()).argmax()
    width_min = (image.sum(axis=0) != 0).argmax()
    width_max = ((image.sum(axis=0) != 0).cumsum()).argmax()
    head_top = image[height_min, :].argmax()
    # �����и��ͼƬ�Ĵ�С��Ϊsize*size����Ϊ�˵ĸ�һ�㶼����ڿ�
    size = height_max - height_min
    temp = np.zeros((size, size))
    # ��width_max-width_min������height_max-height_min���ߣ�szie�����˵�����ͼ������size*size��ͼƬ����
    # l = (width_max-width_min)//2
    # r = width_max-width_min-l
    # ��ͷΪ���ģ�����width_max-width_min������height_max-height_min���ߣ�szie�����˵�����ͼ������size*size��ͼƬ����
    l1 = head_top - width_min
    r1 = width_max - head_top
    # ������ڸߣ���ͷ�������Ҳ����ӱ�Ҫ����ͼƬ��һ��Ҫ�����ͼƬΪ������Ҫ���ͼƬ
    flag = False
    if size <= width_max - width_min or size // 2 < r1 or size // 2 < l1:
        flag = True
        return temp, flag
    temp[:, (size // 2 - l1):(size // 2 + r1)] = image[height_min:height_max, width_min:width_max]
    return temp, flag



if __name__ == '__main__':
    path = '��̬����ͼ�洢·��'
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