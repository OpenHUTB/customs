# Retinaface+arcface：人脸识别模型在Pytorch当中的实现

---
## 目录
1. [注意事项 Attention](#注意事项)
2. [所需环境 Environment](#所需环境)
3. [文件说明 Description](#文件说明)
4. [预测步骤 How2predict](#预测步骤)
5. [参考资料 Reference](#Reference)

## 注意事项
该库中包含了两个网络，分别是arcface和retinaface。二者使用不同的权值。    
在使用网络时一定要注意权值的选择，以及主干与权值的匹配。   
本项目将权重文件转为onnx格式，方便后面部署      

## 所需环境
pytorch==1.2.0 

## 文件说明
### 预测文件说明
权重文件放入model_data文件夹中,其中retinaface_mobilenet0.25.onnx为retinaface的权重文件，   
arcface.onnx为arcface的权重文件   
初次运行时，需要运行encoding.py,将datasets中的图片导入数据库，生成face_encoding.npy和face_name.npy。 
数据库更新可以再次运行encoding()函数，也可以运行encoding_one()函数   

### 项目原理介绍
1、retinaface网络进行人脸检测，在所有人脸中选择面积最大的人脸，将最大人脸crop进行人脸矫正对齐，利用arcface将人脸转为512维的特征向量，作为数据库人脸存储    
2、retinaface网络进行人脸检测，将检测到的人脸crop进行人脸矫正对齐，利用arcface将人脸转为512维的特征向量，与数据库中人脸进行对比    
3、为解决大量高维数据库人脸对比困难的问题，采用faiss为数据库的人脸进行标记索引检索    

## 预测步骤
1. 本项目为retinaface和arcface组成，只包含测试使用部分，可以直接运行。
3. config.py文件里面，可修改参数，适配自己的项目  
```python
cfg_mnet = {
    # ----------------------------------------------------------------------#
    #   图像检测中会将图片resize的大小
    # ----------------------------------------------------------------------#
    'test_image_size'   : 640,
    # ----------------------------------------------------------------------#
    #   retinaface中只有得分大于置信度的预测框会被保留下来
    # ----------------------------------------------------------------------#
    "confidence"        : 0.5,
    # ----------------------------------------------------------------------#
    #   retinaface中非极大抑制所用到的nms_iou大小
    # ----------------------------------------------------------------------#
    "nms_iou"           : 0.3,
    # ----------------------------------------------------------------------#
    #   arcface模型不同人脸的阈值
    # ----------------------------------------------------------------------#
    "facenet_threhold"  : 27.0,
    # ----------------------------------------------------------------------#
    #   faiss索引不同人脸的阈值
    # ----------------------------------------------------------------------#
    "faiss distance"    : 750,
    'device'            : 'cuda:0' if torch.cuda.is_available() else 'cpu'
}
```
3. 运行encoding.py，对face_dataset里面的图片进行编码，face_dataset的命名规则为XXX_1.jpg、XXX_2.jpg。最终在model_data文件夹下生成对应的数据库人脸编码数据文件。
4. 运行predict.py，输入下述文字，可直接预测。
```python
img/img.png
```
<img alt="img.png" height="400" src="img.png" width="400"/>     

5. 在predict.py里面进行设置可以进行video视频检测。  


## Reference
https://github.com/biubug6/Pytorch_Retinaface       
https://zhuanlan.zhihu.com/p/357414033  
https://github.com/bubbliiiing/facenet-retinaface-pytorch   