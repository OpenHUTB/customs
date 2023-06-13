# -*- coding:utf-8 -*-
# author:peng
# Date：2023/4/15 14:38
import cv2
import torch
from torch import nn

# a = torch.ones(4, 128)
# b = torch.ones(4, 128)
# c = torch.zeros(4, 128)
# d = torch.randn(4,128)
# criterion = nn.TripletMarginLoss(margin=0.2)
# loss1 = criterion(a,b,c)
# loss2 = criterion(a,c,b)
# loss3 = criterion(a,d,b)
# loss4 = criterion(a,d,c)
# print(loss1,loss2,loss3,loss4)
#
# import torch
# print(torch.__version__)  #注意是双下划线
video_path = 'img/1683989579163.MP4'
capture = cv2.VideoCapture(video_path)
if video_save_path != "":
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    out = cv2.VideoWriter(video_save_path, fourcc, video_fps, size)

ref, frame = capture.read()
if not ref:
    raise ValueError("未能正确读取摄像头（视频），请注意是否正确安装摄像头（是否正确填写视频路径）。")

fps = 0.0
while True:
    t1 = time.time()
    # 读取某一帧
    ref, frame = capture.read()
    if not ref:
        break
    # 格式转变，BGRtoRGB
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # 进行检测
    frame = retinaface.detect_image(frame)
    # RGBtoBGR满足opencv显示格式
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    fps = (fps + (1. / (time.time() - t1))) / 2
    print("fps= %.2f" % (fps))
    frame = cv2.putText(frame, "fps= %.2f" % (fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("video", frame)
    c = cv2.waitKey(1) & 0xff
    if video_save_path != "":
        out.write(frame)

    if c == 27:
        capture.release()
        break
print("Video Detection Done!")
capture.release()
if video_save_path != "":
    print("Save processed video to the path :" + video_save_path)
    out.release()
cv2.destroyAllWindows()

