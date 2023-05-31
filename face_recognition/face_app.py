import os
import cv2
import numpy as np
import onnxruntime
from tqdm import tqdm
from utils.scrfd import SCRFD
from utils.arcface import ArcFaceONNX
from PIL import Image, ImageDraw, ImageFont
from utils.utils import compare_faces

def cv2ImgAddText(img, label, left, top, textColor=(255, 255, 255)):
    img = Image.fromarray(img)
    #   设置字体
    font = ImageFont.truetype(font='model_data/simhei.ttf', size=20)

    draw = ImageDraw.Draw(img)
    label = label.encode('utf-8')
    draw.text((left, top), str(label,'UTF-8'), fill=textColor, font=font)
    return np.asarray(img)

class FaceRecognition(object):

    _defaults = {
        #   人脸检测网络权重路径
        "det_model_path": 'models/buffalo_l/det_10g.onnx',
        #   人脸识别网络权重路径
        "rec_model_path": 'models/buffalo_l/w600k_r50.onnx',
        #   数据库路径
        "face_embeddings_path": 'model_data/face_db.npy',
        "face_names_path": 'model_data/names.npy',
        #   人脸检测置信度阈值
        "confidence": 0.5,
        #   人脸检测非极大值抑制阈值
        "nms_threshold": 0.3,
        #   人脸识别门限阈值
        "face_threshold": 1.23,
        #   运行设备id，小于0为cpu
        "device_id": 0
    }


    def __init__(self, encoding=0, **kwargs):
        '''

        :param encoding: 0表示正常模式，1表示人脸编码模式
        :param kwargs:
        '''
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)

        self.det_model, self.rec_model = self.load_model()

        try:
            self.known_face_embeddings, self.known_face_names = self.load_faces()
        except:
            if not encoding:
                print("载入已有人脸特征失败，请检查model_data下面是否生成了相关的人脸特征文件。")
                pass

    def load_model(self):

        # 设置 ONNX Runtime 日志的默认级别
        onnxruntime.set_default_logger_severity(3)
        # 加载人脸检测模型
        detector = SCRFD(self.det_model_path)
        detector.prepare(ctx_id=self.device_id)
        # 加载人脸特征提取模型
        rec = ArcFaceONNX(self.rec_model_path)
        rec.prepare(self.device_id)
        # 返回模型对象
        return detector, rec

    def load_faces(self):

        embeddings = np.load(self.face_embeddings_path)
        names = np.load(self.face_names_path)

        return embeddings, names

    #   添加人脸
    def add_face(self, image_path):

        image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), -1)
        name = image_path.split("/")[-1].split('_')[0]
        face, kps = self.det_model.autodetect(image)
        # 判断人脸是否存在
        assert face.shape[0] == 1, "图片中未检测到人脸或超过一张人脸"

        embedding, _ = self.rec_model.get(image, kps[0])

        matches, face_distances = compare_faces(self.known_face_embeddings, embedding, self.face_threshold)

        best_match_index = np.argmin(face_distances)
        name_list = self.known_face_names[best_match_index]

        if name not in name_list:

            self.known_face_embeddings.append(embedding)
            self.known_face_names.append()

            np.save(self.face_embeddings_path, self.known_face_embeddings)
            np.save(self.face_names_path, self.known_face_names)
            print("添加人脸成功")
        else:
            print("数据库中已存在该人脸{}".format(name))

    def face_to_embedding(self, image_paths, sava_face=False):

        embeddings = []
        names = []
        for file in tqdm(os.listdir(image_paths)):
            img = cv2.imdecode(np.fromfile(os.path.join(image_paths, file), dtype=np.uint8), -1)
            name = file.split("_")[0]
            face, kps = self.det_model.autodetect(img)
            if face.shape[0] != 1:
                print("{} 添加失败".format(file))
                continue
            else:
                embedding, face_img = self.rec_model.get(img, kps[0])
                if sava_face:
                    save_face_path = "faces"
                    if os.path.exists(save_face_path):
                        os.makedirs(save_face_path)
                    cv2.imencode('.jpg', face_img)[1].tofile(os.path.join(save_face_path, file))
                embeddings.append(embedding)
                names.append(name)
        print("Finish")

        np.save(self.face_embeddings_path, embeddings)
        np.save(self.face_names_path, names)


    #   人脸识别
    def face_rec(self, image):

        dimg = image.copy()

        bboxes, kpss = self.det_model.autodetect(image)

        # 对每张人脸进行对比
        for i in range(bboxes.shape[0]):
            kps = kpss[i]
            embedding, _ = self.rec_model.get(image, kps)
            matches, face_distances = compare_faces(self.known_face_embeddings, embedding)

            name = "Unknown"
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = self.known_face_names[best_match_index]

            if name == "Unknown":
                color = (0, 0, 255)
            else:
                color = (0, 255, 0)

            box = bboxes[i][:-1].astype(np.int32)
            pro = "{:.2f}".format(bboxes[i][-1])

            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.rectangle(dimg, (box[0], box[1]), (box[2], box[3]), color, 1)
            cv2.putText(dimg, pro, (box[0], box[1] + 12), font, 0.5, (255, 255, 255), 1)
            dimg = cv2ImgAddText(dimg, name, box[0]+5, box[3] - 25)

            kps = kps.astype(np.int32)
            for l in range(kps.shape[0]):
                color = (255, 0, 0)
                if l == 0 or l == 3:
                    color = (0, 255, 0)
                cv2.circle(dimg, (kps[l][0], kps[l][1]), 1, color, 2)

        return dimg
