# -*- coding:utf-8 -*-
# author:peng
# Date：2023/7/3 19:41
import os.path
import time
import faiss
import cv2
import matplotlib.pyplot as plt
import numpy as np

from retinaface_arcface import retinaface_arcface
from utils.utils import letterbox_image, put_box_img

if __name__ == "__main__":
    retinaface = retinaface_arcface()
    # ----------------------------------------------------------------------------------------------------------#
    #   mode用于指定测试的模式：
    #   'predict'表示单张图片预测，如果想对预测过程进行修改，如保存图片，截取对象等，可以先看下方详细的注释
    #   'video'表示视频检测，可调用摄像头或者视频进行检测，详情查看下方注释。
    #   'fps'表示测试fps，使用的图片是img里面的street.jpg，详情查看下方注释。
    #   'dir_predict'表示遍历文件夹进行检测并保存。默认遍历img文件夹，保存img_out文件夹，详情查看下方注释。
    # ----------------------------------------------------------------------------------------------------------#
    mode = "predict"
    # ----------------------------------------------------------------------------------------------------------#
    #   video_path用于指定视频的路径，当video_path=0时表示检测摄像头
    #   想要检测视频，则设置如video_path = "xxx.mp4"即可，代表读取出根目录下的xxx.mp4文件。
    #   video_save_path表示视频保存的路径，当video_save_path=""时表示不保存
    #   想要保存视频，则设置如video_save_path = "yyy.mp4"即可，代表保存为根目录下的yyy.mp4文件。
    #   video_fps用于保存的视频的fps
    #   video_path、video_save_path和video_fps仅在mode='video'时有效
    #   保存视频时需要ctrl+c退出或者运行到最后一帧才会完成完整的保存步骤。
    # ----------------------------------------------------------------------------------------------------------#
    video_path = 'img/peng_mask.mp4'
    video_save_path = "img_out/camera_predict.mp4"
    video_fps = 25.0
    # -------------------------------------------------------------------------#
    #   dir_origin_path指定了用于检测的图片的文件夹路径
    #   dir_save_path指定了检测完图片的保存路径
    #   dir_origin_path和dir_save_path仅在mode='dir_predict'时有效
    # -------------------------------------------------------------------------#
    dir_origin_path = "img/"
    dir_save_path = "img_out/"

    # 加载数据
    known_face_encodings = np.load("model_data/face_encoding.npy")
    known_face_names = np.load("model_data/face_name.npy")
    print('known_face_encodings:', known_face_encodings.shape)
    print('known_face_names:', known_face_names.shape)
    # 利用faiss构建索引进行排序查询
    dim, measure = 512, faiss.METRIC_L2
    # 详细了解参考：https://zhuanlan.zhihu.com/p/357414033
    param = ['Flat', 'IVF100,Flat', 'PQ16', 'IVF100,PQ16', 'LSH', 'HNSW64']
    index = faiss.index_factory(dim, param[0], measure)
    index.add(known_face_encodings)

    if mode == "predict":
        while True:
            img = input('Input image filename:')
            image = cv2.imread(img)
            if image is None:
                print('Open Error! Try again!')
                continue
            else:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                boxes_conf_landms, face_encodings = retinaface.detect_image(image)
                if boxes_conf_landms is not None:
                    print(boxes_conf_landms.shape, len(face_encodings))
                    D, I = index.search(np.array(face_encodings), 1)
                    for i, d in zip(I, D):
                        name = known_face_names[i[0]]
                        print('faiss name:', name)
                        print('faiss distance:', d[0])
                else:
                    continue
                # 画框
                image = put_box_img(image, boxes_conf_landms, index, known_face_names, face_encodings)
                r_image = letterbox_image(image, (800, 800)).astype(np.uint8)
                plt.Figure()
                plt.imshow(r_image)
                plt.show()
                break

    elif mode == "video":
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
            boxes_conf_landms, face_encodings = retinaface.detect_image(frame)
            if boxes_conf_landms is None:
                # RGBtoBGR满足opencv显示格式
                frame = frame[:, :, ::-1]
                cv2.imshow("video", frame)
                c = cv2.waitKey(1) & 0xff
                if video_save_path != "":
                    if not os.path.exists('img_out'):
                        os.mkdir('img_out')
                    out.write(frame)
                if c == 27:
                    capture.release()
                    break
                continue
            # 画框
            frame = put_box_img(frame, boxes_conf_landms, known_face_encodings, known_face_names, face_encodings)

            fps = (fps + (1. / (time.time() - t1))) / 2
            print("fps= %.2f" % (fps))
            frame = cv2.putText(frame, "fps= %.2f" % (fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            # RGBtoBGR满足opencv显示格式
            frame = frame[:, :, ::-1]

            cv2.imshow("video", frame)
            c = cv2.waitKey(1) & 0xff
            if video_save_path != "":
                if not os.path.exists('img_out'):
                    os.mkdir('img_out')
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
