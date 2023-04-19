import cv2
from insightface.app import FaceAnalysis
from tqdm import tqdm
import os
import numpy as np


def face2embedding(img_path: str, save_path: str):
    # 加载模型
    app = FaceAnalysis(root="./", allowed_modules=['detection', 'recognition'])  # enable detection model only
    app.prepare(ctx_id=0, det_size=(640, 640))

    # 读取数据
    image_folder = img_path

    face_db = []
    names = []

    for file in tqdm(os.listdir(image_folder)):
        image_path = os.path.join(image_folder, file)
        try:
            image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        except:
            continue
        # 返回所有人脸检测结果
        faces = app.get(image)
        num_faces = len(faces)
        if num_faces > 1:
            raise ValueError("图像>>{}<<中包含多个人脸，请重新录入。".format(image_path))
        if num_faces < 1:
            continue

        # 对所有脸进行编码
        embeddings = faces[0].normed_embedding
        face_db.append(embeddings)
        names.append(file.split('_')[0])

    np.save(os.path.join(save_path, 'names.npy'), np.array(names))
    np.save(os.path.join(save_path, 'face_db.npy'), np.array(face_db))


if __name__ == '__main__':
    face2embedding(img_path='face_dataset', save_path="model_data")
