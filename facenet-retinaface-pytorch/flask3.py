# -*- coding:utf-8 -*-
# author:peng
# Date：2023/4/27 21:39
import base64

import cv2
import numpy as np
from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit, send

from retinaface import Retinaface
from util.utils import letterbox_image

app = Flask(__name__)
socketio = SocketIO(app)

camera_id = 0  # 摄像头 ID（可以通过试错获取）
cap = cv2.VideoCapture(camera_id)

@socketio.on('start_camera')
def start_camera(a):
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


@app.route('/')
def index():
    return render_template('t2.html')


if __name__ == '__main__':
    socketio.run(app, allow_unsafe_werkzeug=True)
