# -*- coding:utf-8 -*-
# Author:      zhousf
# Date:        2019-09-05
# File:        prediction.py
# Description: 预测
import numpy as np
import os
import cv2
import itertools
from model import license_plate_model


class PredictionModel(object):
    def __init__(self, weight_file):
        self.label_chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 "
        self.letters = [letter for letter in self.label_chars]
        self.img_height = 64
        self.img_width = 128
        self.label_max_length = 9
        self.class_num = len([letter for letter in self.label_chars]) + 1
        self.data_format = 'channels_last'
        self.model = self.load_model(weight_file)

    def load_model(self, weight_file):
        model_ = license_plate_model.Model(self.img_height, self.img_width, self.class_num, self.label_max_length)
        model_ = model_.create_model(training=False, data_format=self.data_format)
        model_.load_weights(weight_file)
        return model_

    def decode_label(self, out):
        # 去掉前两行
        out = out[0, 2:]
        score = list(np.max(out, axis=1))
        # 取得每行最大值的索引
        out_best = list(np.argmax(out, axis=1))
        print(out_best)
        print(score)
        # 分组：相同值归并到一组
        out_best_ = [k for k, g in itertools.groupby(out_best)]
        label = ''
        for index in out_best_:
            if index < len(self.letters):
                label += self.letters[index]
        return label

    def predict_image(self, img):
        """
        预测单张图片
        :param img:
        :return:
        """
        if img.endswith("jpg") \
                or img.endswith(".JPG") \
                or img.endswith(".png"):
            image = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
            image = cv2.resize(image, (self.img_height, self.img_width))
            if self.data_format == 'channels_last':
                image = image.T
                image = np.expand_dims(image, -1)
                image = np.expand_dims(image, axis=0)
            if self.data_format == 'channels_first':
                image = np.expand_dims(image, -1)
                image = np.expand_dims(image, axis=0)
                image = image.T
            net_out_value = self.model.predict(image)
            predict_str = self.decode_label(net_out_value)
            predict_str = predict_str.strip()
            predict_str = predict_str.strip()
            return predict_str
        return None

    def predict_images(self, img_dir):
        """
        批量预测
        :param img_dir:
        :return:
        """
        total = 0
        accuracy = 0
        letter_total = 0
        letter_accuracy = 0
        for root, dirs, files in os.walk(img_dir):
            for img in files:
                predict_str = self.predict_image(os.path.join(root, img))
                predict_str = predict_str.strip()
                plate = img[0:-4]
                plate = plate.upper()
                plate = plate.strip()
                for i in range(min(len(predict_str), len(plate))):
                    if predict_str[i] == plate[i]:
                        letter_accuracy += 1
                letter_total += max(len(predict_str), len(plate))
                success = False
                if predict_str == plate:
                    accuracy += 1
                    success = True
                total += 1
                print('预测: {0} 真值: {1}  {2}'.format(predict_str, plate, success))
        print("正确率 : ", accuracy / total)
        print("字符正确率: ", letter_accuracy / letter_total)


if __name__ == "__main__":
    current_dir = os.path.dirname(__file__)
    weight_file = os.path.join(current_dir, "train_dir/13_0.213.hdf5")
    img_dir = os.path.join(current_dir, "img/pre")
    img_file = os.path.join(img_dir, "15B603WV.jpg")
    model = PredictionModel(weight_file)
    print(model.predict_image(img=img_file))
    # model.predict_images(img_dir=img_dir)



