import cv2
import os

path = '源视频文件路径'
path1 = os.listdir(path)
for name in path1:
    base_name=os.path.splitext(name)[0]
    video_path = path + '/' +name
    # # videos = os.listdir(video_path)
    vc = cv2.VideoCapture(video_path)  # 视频抽帧
    c = 0
    i = 0
    rval = vc.isOpened()
    dir_name1 = "视频帧存储路径"
    if not os.path.isdir(dir_name1):
        os.makedirs(dir_name1)
    count = len(os.listdir(dir_name1))
    dir_name11 = dir_name1 + '/' + f"{count+1:02}"
    if not os.path.isdir(dir_name11):
        os.makedirs(dir_name11)
    while rval:
        c = c + 1
        rval, frame = vc.read()
        if rval:
            # cropImg = frame[frame_id[i].y1:frame_id[i].y2, frame_id[i].x1:frame_id[i].x2]
            num = len(os.listdir(dir_name11))
            dir_name2 = dir_name11 + "/" + str(num + 1) + '.png'
            cv2.imwrite(dir_name2, frame)
            cv2.waitKey(1)
        else:
            break
    vc.release()
print('save_success')

