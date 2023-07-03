## 权重文件链接
链接: https://pan.baidu.com/s/11swWpsDTNAq168ms_ErPBA?pwd=5kpu 提取码: 5kpu 
将下载好的权重zip文件移动到models目录下，直接解压缩。
face_dataset用于存放人脸图片，用于录入人脸数据；
img目录用于存放待识别别的人脸图片或视频，用于测试模型；
img_out目录用于存放识别后的数据；
model_data目录用于存放人脸数据库；
models目录用于存放权重文件；
face_app.py里面可修改各种数据的路径；
face2embedding.py用于将face_dataset里面的人脸数据编码成人脸数据，并将生成的文件保存到model_data目录;
face_recognition.py是main函数，用于测试人脸识别效果。
