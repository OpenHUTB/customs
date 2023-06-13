import os

from retinaface import Retinaface

'''
在更换facenet网络后一定要重新进行人脸编码，运行encoding.py。
'''


def encoding():
    retinaface = Retinaface(1)

    list_dir = os.listdir("face_dataset")
    image_paths = []
    names = []
    for name in list_dir:
        image_paths.append("face_dataset/" + name)
        names.append(name.split(".")[0])
        print(name.split(".")[0])

    retinaface.encode_face_dataset(image_paths, names)


if __name__ == '__main__':
    encoding()