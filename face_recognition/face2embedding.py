import os
import os.path as osp
import cv2
import numpy as np
import onnxruntime
from utils.scrfd import SCRFD
from utils.arcface import ArcFaceONNX
from tqdm import tqdm

onnxruntime.set_default_logger_severity(3)

assets_dir = osp.expanduser('models/buffalo_l')

detector = SCRFD(os.path.join(assets_dir, 'det_10g.onnx'))
detector.prepare(0)
model_path = os.path.join(assets_dir, 'w600k_r50.onnx')
rec = ArcFaceONNX(model_path)
rec.prepare(0)

def face2embedding(image_folder: str, save_path: str):

    # 读取数据
    face_db = []
    names = []

    for file in tqdm(os.listdir(image_folder)):
        image_path = os.path.join(image_folder, file)
        try:
            image = cv2.imread(image_path)
        except:
            continue

        # 返回所有人脸检测结果

        bboxes, kpss = detector.autodetect(image, max_num=1)
        
        assert bboxes.shape[0] == 1, "图像>>{}<<中包含多个人脸，请重新录入。".format(image_path)

        kps = kpss[0]
        # 对所有脸进行编码
        embedding = rec.get(image, kps)
        normalized_embedding = embedding / np.linalg.norm(embedding)
        face_db.append(normalized_embedding)
        names.append(file.split('_')[0])

    np.save(os.path.join(save_path, 'names.npy'), np.array(names))
    np.save(os.path.join(save_path, 'face_db.npy'), np.array(face_db))

if __name__ == '__main__':
    face2embedding(image_folder='face_dataset', save_path="model_data")
