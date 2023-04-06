# -*- coding:utf-8 -*-
# author:peng
# Date：2023/4/4 11:50
import time

import cv2
import numpy as np
from flask import request, Flask, render_template
from flask_cors import CORS

from Mtcnn_interface import mtcnn_detector, face_recognition, list_to_json

app = Flask(__name__)
# 允许跨越访问
CORS(app)


@app.route("/recognition", methods=['POST'])
def recognition():
    start_time1 = time.time()
    upload_file = request.files['image']
    if upload_file:
        try:
            img = cv2.imdecode(np.frombuffer(upload_file.read(), np.uint8), 1)
        except:
            return str({"error": 2, "msg": "this file is not image"})
        try:
            info_name, dis, info_bbox, info_landmarks = face_recognition(img)
            if len(info_name) == 0:
                return str({"error": 3, "msg": "image not have face"})
        except:
            return str({"error": 3, "msg": "image not have face"})
        # 封装识别结果
        data_faces = []
        for i in range(len(info_name)):
            data_faces.append(
                {"name": info_name[i], "distance": dis[i],
                 "bbox": list_to_json(np.around(info_bbox[i], decimals=2).tolist()),
                 "landmarks": list_to_json(np.around(info_landmarks[i], decimals=2).tolist())})
        data = str({"code": 0, "msg": "success", "data": data_faces}).replace("'", '"')
        print('duration:[%.0fms]' % ((time.time() - start_time1) * 1000), data)
        return data
    else:
        return str({"error": 1, "msg": "file is None"})


# 注册人脸接口
@app.route("/register", methods=['POST'])
def register():
    global faces_db
    upload_file = request.files['image']
    user_name = request.values.get("name")
    if upload_file:
        try:
            image = cv2.imdecode(np.frombuffer(upload_file.read(), np.uint8), cv2.IMREAD_UNCHANGED)
            boxes, probs, landmarks = mtcnn_detector.detect(image, landmarks=True)
            if probs.shape[0] is not 0:
                if probs.shape[0] == 1:
                    mtcnn_detector(image, save_path='./face_save/{}.jpg'.format(user_name))
                    return str({"code": 0, "msg": "success"})
            return str({"code": 3, "msg": "image not or much face"})
        except:
            return str({"code": 2, "msg": "this file is not image or not face"})
    else:
        return str({"code": 1, "msg": "file is None"})


@app.route('/')
def home():
    return render_template("index.html")


if __name__ == '__main__':
    app.run(host='localhost', port=5000)
