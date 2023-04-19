import cv2
from insightface.app import FaceAnalysis
import numpy as np
import time

# 加载模型
app = FaceAnalysis(root="./", allowed_modules=['detection', 'recognition'])  # enable detection model only
app.prepare(ctx_id=0, det_size=(640, 640))

# 加载人脸库文件和名字列表文件
names = np.load('model_data/names.npy', allow_pickle=True)
face_db = np.load('model_data/face_db.npy', allow_pickle=True)

#   比较人脸
def compare_faces(known_face_encodings, face_encoding_to_check, tolerance=1):

    if len(known_face_encodings) == 0:
        return np.empty((0))
    dis = np.linalg.norm(known_face_encodings - face_encoding_to_check, axis=1)

    return list(dis <= tolerance), dis

def face_recognition(image):

    dimg = image.copy()
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    faces = app.get(img_rgb)
    font = cv2.FONT_HERSHEY_SIMPLEX

    for face in faces:
        embedding = face.normed_embedding
        matches, face_distances = compare_faces(face_db, embedding)
        name = "Unknown"
        #   取出这个最近人脸的评分
        #   取出当前输入进来的人脸，最接近的已知人脸的序号
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = names[best_match_index]

        box = face.bbox.astype(int)
        pro = "{:.4f}".format(face.det_score)

        if name =="Unknown":
            color = (0, 0, 255)
        else:
            color = (0, 255, 0)

        cv2.rectangle(dimg, (box[0], box[1]), (box[2], box[3]), color, 1)
        cv2.putText(dimg, name, (box[0], box[3] - 12), font, 0.5, (255, 255, 255), 1)
        cv2.putText(dimg, pro, (box[0], box[1] + 12), font, 0.5, (255, 255, 255), 1)

        if face.kps is not None:
            kps = face.kps.astype(int)
            # print(landmark.shape)
            for l in range(kps.shape[0]):
                color = (255, 0, 0)
                if l == 0 or l == 3:
                    color = (0, 255, 0)
                cv2.circle(dimg, (kps[l][0], kps[l][1]), 1, color, 2)

    return dimg

if __name__ == "__main__":

    # ----------------------------------------------------------------------------------------------------------#
    #   mode用于指定测试的模式：
    #   'predict'表示单张图片预测，如果想对预测过程进行修改，如保存图片，截取对象等，可以先看下方详细的注释
    #   'video'表示视频检测，可调用摄像头或者视频进行检测，详情查看下方注释。
    #   'dir_predict'表示遍历文件夹进行检测并保存。默认遍历img文件夹，保存img_out文件夹，详情查看下方注释。
    # ----------------------------------------------------------------------------------------------------------#
    mode = "video"
    # ----------------------------------------------------------------------------------------------------------#
    #   video_path用于指定视频的路径，当video_path=0时表示检测摄像头
    #   想要检测视频，则设置如video_path = "xxx.mp4"即可，代表读取出根目录下的xxx.mp4文件。
    #   video_save_path表示视频保存的路径，当video_save_path=""时表示不保存
    #   想要保存视频，则设置如video_save_path = "yyy.mp4"即可，代表保存为根目录下的yyy.mp4文件。
    #   video_fps用于保存的视频的fps
    #   video_path、video_save_path和video_fps仅在mode='video'时有效
    #   保存视频时需要ctrl+c退出或者运行到最后一帧才会完成完整的保存步骤。
    # ----------------------------------------------------------------------------------------------------------#
    video_path = r"C:\Users\97205\Desktop\mp4\pedestrian street3.mp4"
    video_save_path = r"out-2.mp4"
    video_fps = 25.0
    # -------------------------------------------------------------------------#
    #   test_interval用于指定测量fps的时候，图片检测的次数
    #   理论上test_interval越大，fps越准确。
    # -------------------------------------------------------------------------#
    test_interval = 100
    # -------------------------------------------------------------------------#
    #   dir_origin_path指定了用于检测的图片的文件夹路径
    #   dir_save_path指定了检测完图片的保存路径
    #   dir_origin_path和dir_save_path仅在mode='dir_predict'时有效
    # -------------------------------------------------------------------------#
    dir_origin_path = "img/"
    dir_save_path = "img_out/"

    if mode == "predict":

        while True:
            img = input('Input image filename:')
            image = cv2.imread(img)
            if image is None:
                print('Open Error! Try again!')
                continue
            else:

                image = face_recognition(image)

                # r_image = cv2.cvtColor(r_image, cv2.COLOR_RGB2BGR)
                cv2.imshow("after", image)
                cv2.waitKey(0)

    elif mode == "video":
        capture = cv2.VideoCapture(video_path)
        if video_save_path != "":
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            out = cv2.VideoWriter(video_save_path, fourcc, video_fps, size)

        ref, frame = capture.read()
        if not ref:
            raise ValueError("未能正确读取摄像头（视频），请注意是否正确安装摄像头（是否正确填写视频路径）。")

        fps = 0.0
        while (True):
            t1 = time.time()
            # 读取某一帧
            ref, frame = capture.read()
            if not ref:
                break
            frame = face_recognition(frame)

            fps = (fps + (1. / (time.time() - t1))) / 2
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

    elif mode == "dir_predict":
        import os

        from tqdm import tqdm

        img_names = os.listdir(dir_origin_path)
        for img_name in tqdm(img_names):
            if img_name.lower().endswith(
                    ('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
                image_path = os.path.join(dir_origin_path, img_name)
                image = cv2.imread(image_path)
                image = face_recognition(image)
                if not os.path.exists(dir_save_path):
                    os.makedirs(dir_save_path)
                cv2.imwrite(os.path.join(dir_save_path, img_name), image)
    else:
        raise AssertionError("Please specify the correct mode: 'predict', 'video', 'fps' or 'dir_predict'.")