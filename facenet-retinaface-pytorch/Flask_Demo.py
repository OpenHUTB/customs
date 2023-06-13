# -*- coding:utf-8 -*-
# author:peng
# Date：2023/4/21 16:54
import base64
import shutil
from io import BytesIO

import cv2
import numpy as np
from PIL import Image
from flask import Flask, request, render_template
from flask_socketio import SocketIO, send

from encoding import encoding
from retinaface import Retinaface
from util.utils import letterbox_image

app = Flask(__name__, template_folder='templates', static_url_path='', static_folder='')
socketio = SocketIO(app)


@app.route('/')
def index():
    return render_template('tmp.html')


camera_id = 0  # 摄像头 ID（可以通过试错获取）
cap = cv2.VideoCapture(camera_id)


@socketio.on('start_camera')
def start_camera(a):
    print('开启摄像头')
    while True:
        ret, frame = cap.read()
        retinaface = Retinaface()
        r_image = retinaface.detect_image(frame)
        r_image = letterbox_image(r_image, (800, 800)).astype(np.uint8)
        # 处理摄像头数据
        retval, buffer = cv2.imencode('.jpg', r_image)
        jpg_as_text = base64.b64encode(buffer).decode('utf-8')
        # 发送摄像头数据给前端
        send(jpg_as_text, broadcast=True)


@socketio.on('stop_stream')
def stop_stream(a):
    cap.release()  # 关闭摄像头
    print('关闭摄像头')
    cv2.destroyAllWindows()


@app.route("/upload", methods=['POST'])
def upload():
    print(request.files['image'])
    # 获取图片数据
    image_data = request.files['image'].read()
    # 图片保存
    image = Image.open(BytesIO(image_data)).convert('RGB')
    image.save('image.jpg')
    # 图片分析与上传
    retinaface = Retinaface()
    image = np.array(image)
    r_image = retinaface.detect_image(image)
    r_image = cv2.cvtColor(r_image, cv2.COLOR_RGB2BGR)
    r_image = letterbox_image(r_image, (800, 800)).astype(np.uint8)
    img_str = cv2.imencode('.jpg', r_image)[1].tobytes()
    image = base64.b64encode(img_str).decode('utf-8')
    # print(image)
    return image


@app.route("/save", methods=['GET', 'POST'])
def save():
    value = request.form['name']
    shutil.copy('image.jpg', './face_dataset/{}.jpg'.format(value))
    # print(value)
    encoding()


if __name__ == "__main__":
    socketio.run(app, allow_unsafe_werkzeug=True)
