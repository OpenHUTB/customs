import torch
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
import os
from tqdm import tqdm
import numpy as np

def face2embedding(image_path:str, save_path:str):

    # 加载人脸检测和特征提取模型
    # MTCNN为人脸检测模型，resnet为人脸特征提取模型
    mtcnn = MTCNN(keep_all=True)
    resnet = InceptionResnetV1(pretrained='vggface2').eval()
    # 设置图片文件夹路径和保存文件名
    image_folder = image_path
    save_folder = save_path

    # 如果保存文件夹不存在，则创建一个新文件夹
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # 设置保存文件名
    save_file = os.path.join(save_folder, 'face_db.pt')
    # 设置保存.npy文件名和文本文件名
    # save_file = os.path.join(save_folder, 'face_db.npy')
    txt_file = os.path.join(save_folder, 'names.npy')

    # 初始化人脸库列表和名字列表
    face_db = []
    names = []

    # 遍历图片文件夹，进行人脸检测和特征提取
    for file in tqdm(os.listdir(image_folder)):
        image_path = os.path.join(image_folder, file)
        try:
            image = Image.open(image_path).convert('RGB')
        except:
            continue
        # 返回所有脸的Tensor
        faces = mtcnn(image)
        num_faces = len(faces)
        if num_faces > 1:
            raise ValueError("图像>>{}<<中包含多个人脸，请重新录入。".format(image_path))
        # 对所有脸进行编码
        embeddings = resnet(faces)
        embeddings = embeddings / embeddings.norm(2, dim=1, keepdim=True)
        if len(embeddings) > 0:
            face_db.append(embeddings.mean(dim=0))
            names.append(file.split('_')[0])

    # 将人脸库保存到文件中
    torch.save(torch.stack(face_db), save_file)
    # 将名字列表保存为.npy文件
    np.save(txt_file, np.array(names))

if __name__=="__main__":

    # 对要录入的人脸数据进行embedding，每张图片中只能包含一张人脸，图片命名格式为name_index.jpg
    face2embedding(image_path = 'face_dataset', save_path = 'model_data')




