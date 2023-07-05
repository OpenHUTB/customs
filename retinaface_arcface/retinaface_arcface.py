# -*- coding:utf-8 -*-
# author:peng
# Date：2023/7/3 19:14
import numpy as np
import onnxruntime
import torch
from PIL import Image
from tqdm import tqdm

from utils.Anchors import Anchors
from utils.config import cfg_mnet
from utils.utils import preprocess_input, letterbox_image, decode, decode_landm, non_max_suppression, \
    retinaface_correct_boxes, Alignment_1


class retinaface_arcface:
    def __init__(self, retinaface_path='model_data/retinaface_mobilenet0.25.onnx',
                 arcface_path='model_data/arcface.onnx'):
        self.cfg = cfg_mnet
        self.anchors = Anchors(self.cfg,
                               image_size=(self.cfg['test_image_size'], self.cfg['test_image_size'])).get_anchors()

        # 加载模型
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if self.cfg['device'] != 'cpu' else ['CPUExecutionProvider']
        self.retinaface = onnxruntime.InferenceSession(retinaface_path, providers=providers)
        self.arcface = onnxruntime.InferenceSession(arcface_path, providers=providers)

    def detect_image(self, image):
        '''
        image: numpy RGB
        '''
        old_image = image.copy()
        image = preprocess_input(np.array(image, np.float32))
        im_height, im_width, _ = np.shape(image)

        scale = [
            np.shape(image)[1], np.shape(image)[0], np.shape(image)[1], np.shape(image)[0]
        ]
        scale_for_landmarks = [
            np.shape(image)[1], np.shape(image)[0], np.shape(image)[1], np.shape(image)[0],
            np.shape(image)[1], np.shape(image)[0], np.shape(image)[1], np.shape(image)[0],
            np.shape(image)[1], np.shape(image)[0]
        ]
        # 不失真的resize
        image = letterbox_image(image, (self.cfg['test_image_size'], self.cfg['test_image_size']))
        photo = np.expand_dims(np.transpose(np.array(image, np.float32), (2, 0, 1)), 0)

        # 人脸检测
        retinaface_input_name = self.retinaface.get_inputs()[0].name
        outputs = self.retinaface.run(None, {retinaface_input_name: photo})
        loc = torch.tensor(outputs[0])
        conf = torch.tensor(outputs[1])
        landms = torch.tensor(outputs[2])
        # ---------------------------------------------------#
        #   Retinaface网络的解码，最终我们会获得预测框
        #   将预测结果进行解码和非极大抑制
        # ---------------------------------------------------#
        boxes = decode(loc.data.squeeze(0), self.anchors, self.cfg['variance'])
        conf = conf.data.squeeze(0)[:, 1:2]
        landms = decode_landm(landms.data.squeeze(0), self.anchors, self.cfg['variance'])

        # -----------------------------------------------------------#
        #   对人脸检测结果进行堆叠
        # -----------------------------------------------------------#
        boxes_conf_landms = torch.cat([boxes, conf, landms], -1)
        boxes_conf_landms = non_max_suppression(boxes_conf_landms, self.cfg['confidence'], self.cfg['nms_iou'])

        if len(boxes_conf_landms) <= 0:
            print('no' * 20)
            return None, None

        # 去除灰度条
        boxes_conf_landms = retinaface_correct_boxes(boxes_conf_landms, np.array(
            [self.cfg['test_image_size'], self.cfg['test_image_size']]),
                                                     np.array([im_height, im_width]))


        boxes_conf_landms[:, :4] = boxes_conf_landms[:, :4] * scale
        boxes_conf_landms[:, 5:] = boxes_conf_landms[:, 5:] * scale_for_landmarks

        # 人脸识别
        face_encodings = []
        for boxes_conf_landm in boxes_conf_landms:
            # ----------------------#
            #   图像截取，人脸矫正
            # ----------------------#
            boxes_conf_landm = np.maximum(boxes_conf_landm, 0)  # 返回大于0的数
            crop_img = np.array(old_image)[int(boxes_conf_landm[1]):int(boxes_conf_landm[3]),
                       int(boxes_conf_landm[0]):int(boxes_conf_landm[2])]
            landmark = np.reshape(boxes_conf_landm[5:], (5, 2)) - np.array(
                [int(boxes_conf_landm[0]), int(boxes_conf_landm[1])])
            crop_img, _ = Alignment_1(crop_img, landmark)

            crop_img = np.array(letterbox_image(crop_img, (112, 112)), np.float32) / 255
            crop_img = np.expand_dims(crop_img.transpose(2, 0, 1), 0)

            arcface_output_name = self.arcface.get_outputs()[0].name
            arcface_input_name = self.arcface.get_inputs()[0].name
            arcface_feature = self.arcface.run([arcface_output_name], {arcface_input_name: crop_img})[0]
            arcface_feature = np.reshape(arcface_feature, 512)
            face_encodings.append(arcface_feature)
        return boxes_conf_landms, face_encodings

    def encode_face_dataset(self, image_paths, names):
        '''
        image_paths: list
        names: list
        '''
        face_encodings = []
        name_encodings = []
        for index, path in enumerate(tqdm(image_paths)):
            image = Image.open(path).convert('RGB')
            old_image = np.array(image).copy()
            image = preprocess_input(np.array(image, np.float32))
            im_height, im_width, _ = np.shape(image)

            scale = [
                np.shape(image)[1], np.shape(image)[0], np.shape(image)[1], np.shape(image)[0]
            ]
            scale_for_landmarks = [
                np.shape(image)[1], np.shape(image)[0], np.shape(image)[1], np.shape(image)[0],
                np.shape(image)[1], np.shape(image)[0], np.shape(image)[1], np.shape(image)[0],
                np.shape(image)[1], np.shape(image)[0]
            ]
            # 不失真的resize
            image = letterbox_image(image, (self.cfg['test_image_size'], self.cfg['test_image_size']))
            photo = np.expand_dims(np.transpose(np.array(image, np.float32), (2, 0, 1)), 0)

            # 人脸检测
            retinaface_input_name = self.retinaface.get_inputs()[0].name
            outputs = self.retinaface.run(None, {retinaface_input_name: photo})
            loc = torch.tensor(outputs[0])
            conf = torch.tensor(outputs[1])
            landms = torch.tensor(outputs[2])
            # ---------------------------------------------------#
            #   Retinaface网络的解码，最终我们会获得预测框
            #   将预测结果进行解码和非极大抑制
            # ---------------------------------------------------#
            boxes = decode(loc.data.squeeze(0), self.anchors, self.cfg['variance'])
            conf = conf.data.squeeze(0)[:, 1:2]
            landms = decode_landm(landms.data.squeeze(0), self.anchors, self.cfg['variance'])

            # -----------------------------------------------------------#
            #   对人脸检测结果进行堆叠
            # -----------------------------------------------------------#
            boxes_conf_landms = torch.cat([boxes, conf, landms], -1)
            boxes_conf_landms = non_max_suppression(boxes_conf_landms, self.cfg['confidence'], self.cfg['nms_iou'])

            # 去除灰度条
            boxes_conf_landms = retinaface_correct_boxes(boxes_conf_landms, np.array(
                [self.cfg['test_image_size'], self.cfg['test_image_size']]),
                                                         np.array([im_height, im_width]))

            if len(boxes_conf_landms) <= 0:
                print('no' * 10, names[index], '未检查到人脸', 'no' * 10)
                continue

            boxes_conf_landms[:, :4] = boxes_conf_landms[:, :4] * scale
            boxes_conf_landms[:, 5:] = boxes_conf_landms[:, 5:] * scale_for_landmarks

            # 人脸识别
            # ---------------------------------------------------#
            #   选取最大的人脸框。
            # ---------------------------------------------------#
            best_face_location = None
            biggest_area = 0
            for result in boxes_conf_landms:
                left, top, right, bottom = result[0:4]

                w = right - left
                h = bottom - top
                if w * h > biggest_area:
                    biggest_area = w * h
                    best_face_location = result

            # ----------------------#
            #   图像截取，人脸矫正  #
            # ----------------------#
            crop_img = old_image[int(best_face_location[1]):int(best_face_location[3]),
                       int(best_face_location[0]):int(best_face_location[2])]
            landmark = np.reshape(best_face_location[5:], (5, 2)) - np.array(
                [int(best_face_location[0]), int(best_face_location[1])])
            crop_img, _ = Alignment_1(crop_img, landmark)

            # 人脸resize
            crop_img = np.array(letterbox_image(crop_img, (112, 112)), np.float32) / 255
            crop_img = np.expand_dims(crop_img.transpose(2, 0, 1), 0)

            arcface_output_name = self.arcface.get_outputs()[0].name
            arcface_input_name = self.arcface.get_inputs()[0].name
            arcface_feature = self.arcface.run([arcface_output_name], {arcface_input_name: crop_img})[0]
            arcface_feature = np.reshape(arcface_feature, 512)
            face_encodings.append(arcface_feature)
            name_encodings.append(names[index])

        np.save('model_data/face_encoding.npy', face_encodings)
        np.save('model_data/face_name.npy', name_encodings)