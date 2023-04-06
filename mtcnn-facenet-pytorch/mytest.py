# -*- coding:utf-8 -*-
# author:peng
# Date：2023/4/4 15:47
# 摄像头检测
import cv2
from PIL import Image
from facenet_pytorch import MTCNN
from glob import glob

from Mtcnn_interface import getFacenet
from utils import align


def recognize(image):
    new_frame = None
    mtcnn = MTCNN(margin=20, keep_all=True, device='cpu')
    face_model = getFacenet()
    boxes, probs, landmarks = mtcnn.detect(image, landmarks=True)
    # new_frame, new_landmarks, new_box = align.Alignment_1(frame, landmarks[0], boxes[0])
    dis = []
    info_name = []
    if boxes is not None:
        save_pathes = glob('./face_save/*.jpg')
        for landmark, box in zip(landmarks,boxes):
            new_frame,new_landmark,new_box = align.Alignment_1(image,landmark,box)

            x1, y1, x2, y2 = int(new_box[0]), int(new_box[1]), int(new_box[2]), int(new_box[3])
            img_cut = image[x1:x2, y1:y2]

            name = 'unknown'
            try:
                face = Image.fromarray(cv2.cvtColor(img_cut, cv2.COLOR_BGR2RGB))
                temp_dict = {}
                for save_path in save_pathes:
                    save_img = Image.open(save_path).convert('RGB')
                    l1 = face_model.detect_image(face, save_img)
                    filename = save_path.split('\\')[-1]
                    temp_dict[filename[:-4]] = l1
                dict = sorted(temp_dict.items(), key=lambda x: x[1])  # 按照l2从小到大排序
                if dict[0][1] < 1.1:
                    name = dict[0][0]
                    dis.append(dict[0][1])
                    info_name.append(name)
                else:
                    dis.append(dict[0][1])
                    info_name.append(name)
            except:
                continue

            # cv2.rectangle(new_frame, (x1, y1), (x2, y2), (0, 0, 255), 6)
            cv2.putText(new_frame, name, (x1, y1 - 15), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
            for item in new_landmark:
                x, y = item
                cv2.circle(new_frame, (int(x), int(y)), 1, (0, 255, 0), 4)

        # for box, mark,name in zip(boxes, landmarks,info_name):
        #     x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        #     cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 6)
        #     label = "{}".format(name)
        #
        #     cv2.putText(image, label, (x1, y1 - 15), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        #     for item in mark:
        #         x, y = item
        #         cv2.circle(image, (int(x), int(y)), 1, (0, 255, 0), 4)
    if new_frame is not None:
        return new_frame
    else:
        return image


if __name__ == '__main__':
    # capture = cv2.VideoCapture('./img/aagfhgtpmv.mp4')
    capture = cv2.VideoCapture(0)
    while True:
        ret, draw = capture.read()
        if ret:
            frame = recognize(draw)
            cv2.imshow('video', frame)
            if cv2.waitKey(1) & 0xFF == ord(' '):
                print(type(draw))
                break
        else:
            break

    capture.release()
    cv2.destroyAllWindows()
