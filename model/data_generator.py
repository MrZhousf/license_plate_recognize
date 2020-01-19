# -*- coding:utf-8 -*-
# Author:      zhousf
# Date:        2019-09-03
# File:        DataGenerator.py
# Description:  数据生成器
import tensorflow as tf
import numpy as np
import math
import cv2
import os


class ImageGenerator(tf.keras.utils.Sequence):
    def __init__(self,
                 data_format,
                 label_chars,
                 img_dir,
                 img_height,
                 img_width,
                 label_max_length,
                 batch_size):
        self.data_format = data_format
        self.chars = [char for char in label_chars]
        self.class_num = len([letter for letter in self.chars]) + 1
        self.img_height = img_height
        self.img_width = img_width
        self.batch_size = batch_size
        self.label_max_length = label_max_length
        self.shuffle = True
        self.image = self.read_data(img_dir)
        self.indexes = np.arange(len(self.image))
        self.img_num = len(self.image)
        self.steps_per_epoch = math.ceil(self.img_num / self.batch_size)

    def index_to_label(self, index):
        return "".join(list(map(lambda x: self.chars[int(x)], index)))

    def label_to_index(self, label):
        return list(map(lambda x: self.chars.index(x), label))

    def __len__(self):
        """
        每个epoch迭代次数
        :return:
        """
        return math.ceil(self.img_num / self.batch_size)

    def __getitem__(self, index):
        """
        每个batch数据生成
        :param index:
        :return:
        """
        batch_index = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        # 当批处理索引是最后一个时获取的数据个数不等于batch_size
        if len(batch_index) != self.batch_size:
            batch_index = self.indexes[-self.batch_size:]
        batch_data = [self.image[k] for k in batch_index]
        x, y = self.data_generation(batch_data)
        return x, y

    def on_epoch_end(self):
        """
        每个epoch结束后执行
        :return:
        """
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def data_generation(self, batch_data):
        X_data = np.ones([self.batch_size, self.img_height, self.img_width, 1])
        if self.data_format == 'channels_first':
            X_data = np.ones([self.batch_size, 1, self.img_height, self.img_width])
        Y_data = np.ones([self.batch_size, self.label_max_length])
        input_length = np.ones((self.batch_size, 1)) * (self.img_width // 4 - 2)
        label_length = np.zeros((self.batch_size, 1))
        for i, img in enumerate(batch_data):
            image = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
            image = cv2.resize(image, (self.img_height, self.img_width))
            if self.data_format == 'channels_last':
                image = image.T
                image = np.expand_dims(image, -1)
            if self.data_format == 'channels_first':
                image = np.expand_dims(image, -1)
                image = image.T
            X_data[i] = image
            img = os.path.basename(img)
            label = img[:-9]
            label = label.ljust(self.label_max_length, ' ')
            Y_data[i] = self.label_to_index(label)
            label_length[i] = len(label)
        inputs = {
            'input': X_data,
            'labels': Y_data,
            'input_length': input_length,
            'label_length': label_length
        }
        outputs = {'ctc': np.zeros([self.batch_size])}
        return inputs, outputs

    @staticmethod
    def read_data(img_dir):
        data = []
        for root, dirs, files in os.walk(img_dir):
            for img in files:
                if img.endswith("jpg") \
                        or img.endswith(".JPG") \
                        or img.endswith(".png"):
                    data.append(os.path.join(root, img))
        return data
