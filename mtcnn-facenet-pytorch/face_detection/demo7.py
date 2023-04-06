# -*- coding:utf-8 -*-
# author:peng
# Date：2023/4/4 15:47
# 摄像头检测
import cv2
from PIL import Image
from facenet_pytorch import MTCNN
from glob import glob

from Mtcnn_interface import getFacenet


def recognize(image):
    mtcnn = MTCNN(keep_all=True, device='cpu')
    face_model = getFacenet()
    boxes, probs, landmarks = mtcnn.detect(image, landmarks=True)
    dis = []
    info_name = []
    if boxes is not None:
        save_pathes = glob('./face_save/*.jpg')
        for box in boxes:
            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            img_cut = image[x1:x2, y1:y2]
            face = Image.fromarray(cv2.cvtColor(img_cut,cv2.COLOR_BGR2RGB))
            temp_dict = {}
            for save_path in save_pathes:
                save_img = Image.open(save_path).convert('RGB')
                l1 = face_model.detect_image(face, save_img)
                filename = save_path.split('\\')[-1]
                temp_dict[filename[:-4]] = l1
            dict = sorted(temp_dict.items(), key=lambda x: x[1])  # 按照l2从小到大排序
            if dict[0][1] < 0.8:
                name = dict[0][0]
                dis.append(dict[0][1])
                info_name.append(name)
            else:
                dis.append(dict[0][1])
                info_name.append('unknown')

        k = 0
        for box, mark in zip(boxes, landmarks):
            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 6)
            distance = '%.2f' % dis[k]
            label = "{}, {}".format(info_name[k], distance)
            k+=1
            cv2.putText(image, label, (x1, y1 - 15), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255, 0, 0), 4, cv2.LINE_AA)
            for item in mark:
                x, y = item
                cv2.circle(image, (int(x), int(y)), 1, (0, 255, 0), 4)
    return image


if __name__ == '__main__':
    capture = cv2.VideoCapture(0)
    while True:
        ret, draw = capture.read()
        recognize(draw)
        cv2.imshow('video', draw)
        if cv2.waitKey(1) & 0xFF == ord(' '):
            print(type(draw))
            break

    capture.release()
    cv2.destroyAllWindows()
