# -*- coding:utf-8 -*-
# Author:      zhousf
# Date:        2019-07-25
# File:        license_plate.py
# Description:  CNN + RNN 网络模型
import tensorflow as tf


class Model(object):
    def __init__(self,
                 image_height,
                 image_width,
                 class_num,
                 label_max_length,
                 learning_rate=1e-4):
        self.image_height = image_height
        self.image_width = image_width
        self.channels = 1
        self.class_num = class_num
        self.label_max_length = label_max_length
        self.learning_rate = learning_rate
        self.layer = tf.keras.layers
        self.backend = tf.keras.backend

    def build_cnn(self, inputs, data_format):
        """
        vgg16
        :param inputs: self.layer.Input(shape=(64, 128, 1), dtype=tf.float32)
        :param data_format: channel_last
        :return:
        """
        """
        layer1 (64, 128, 64)
        """
        vgg = self.layer.Conv2D(filters=64,
                                kernel_size=(3, 3),
                                padding='same',
                                data_format=data_format,
                                name='conv1')(inputs)
        vgg = self.layer.BatchNormalization()(vgg)
        vgg = self.layer.Activation('relu')(vgg)
        # (64, 128, 64)
        vgg = self.layer.Conv2D(filters=64,
                                kernel_size=(3, 3),
                                padding='same',
                                data_format=data_format,
                                activation=tf.nn.relu,
                                name='conv2')(vgg)
        vgg = self.layer.BatchNormalization()(vgg)
        vgg = self.layer.Activation('relu')(vgg)
        # (32, 64, 64)
        vgg = self.layer.MaxPool2D(pool_size=(2, 2),
                                   strides=(2, 2),
                                   padding='same',
                                   data_format=data_format,
                                   name='pool1')(vgg)

        """
        layer2 (32, 64, 128)
        """
        vgg = self.layer.Conv2D(filters=128,
                                kernel_size=(3, 3),
                                padding='same',
                                data_format=data_format,
                                name='conv3')(vgg)
        vgg = self.layer.BatchNormalization()(vgg)
        vgg = self.layer.Activation('relu')(vgg)
        # (32, 64, 128)
        vgg = self.layer.Conv2D(filters=128,
                                kernel_size=(3, 3),
                                padding='same',
                                data_format=data_format,
                                activation=tf.nn.relu,
                                name='conv4')(vgg)
        vgg = self.layer.BatchNormalization()(vgg)
        vgg = self.layer.Activation('relu')(vgg)
        # (16, 32, 128)
        vgg = self.layer.MaxPool2D(pool_size=(2, 2),
                                   strides=(2, 2),
                                   padding='same',
                                   data_format=data_format,
                                   name='pool2')(vgg)
        """
        layer3 (16, 32, 256)
        """
        vgg = self.layer.Conv2D(filters=256,
                                kernel_size=(3, 3),
                                padding='same',
                                data_format=data_format,
                                name='conv5')(vgg)
        vgg = self.layer.BatchNormalization()(vgg)
        vgg = self.layer.Activation('relu')(vgg)
        # (16, 32, 256)
        vgg = self.layer.Conv2D(filters=256,
                                kernel_size=(3, 3),
                                padding='same',
                                data_format=data_format,
                                name='conv6')(vgg)
        vgg = self.layer.BatchNormalization()(vgg)
        vgg = self.layer.Activation('relu')(vgg)
        # (8, 32, 256)
        vgg = self.layer.MaxPool2D(pool_size=(2, 2),
                                   strides=(2, 1),
                                   padding='same',
                                   data_format=data_format,
                                   name='pool3')(vgg)
        """
        layer4 (8, 32, 512)
        """
        vgg = self.layer.Conv2D(filters=512,
                                kernel_size=(3, 3),
                                padding='same',
                                data_format=data_format,
                                name='conv7')(vgg)
        vgg = self.layer.BatchNormalization()(vgg)
        vgg = self.layer.Activation('relu')(vgg)
        # (8, 32, 512)
        vgg = self.layer.Conv2D(filters=512,
                                kernel_size=(3, 3),
                                padding='same',
                                data_format=data_format,
                                name='conv8')(vgg)
        vgg = self.layer.BatchNormalization()(vgg)
        vgg = self.layer.Activation('relu')(vgg)
        # (4, 32, 512)
        vgg = self.layer.MaxPool2D(pool_size=(2, 2),
                                   strides=(2, 1),
                                   padding='same',
                                   data_format=data_format,
                                   name='pool4')(vgg)
        """
        layer5 (4, 32, 512)
        """
        vgg = self.layer.Conv2D(filters=512,
                                kernel_size=(3, 3),
                                padding='same',
                                data_format=data_format,
                                name='conv9')(vgg)
        vgg = self.layer.BatchNormalization()(vgg)
        vgg = self.layer.Activation('relu')(vgg)
        # (4, 32, 512)
        vgg = self.layer.Conv2D(filters=512,
                                kernel_size=(3, 3),
                                padding='same',
                                data_format=data_format,
                                activation=tf.nn.relu,
                                name='conv10')(vgg)
        vgg = self.layer.BatchNormalization()(vgg)
        vgg = self.layer.Activation('relu')(vgg)
        # (32, 2048)
        vgg = self.layer.Reshape(target_shape=(32, 2048), name='reshape')(vgg)
        # (32, 64)
        vgg = self.layer.Dense(64, activation=tf.nn.relu, name='dense1')(vgg)
        return vgg

    def build_rnn(self, tensor):
        """
        双向LSTM
        :param tensor: shape=(32, 64)
        :return:
        """
        tensor = self.layer.Bidirectional(self.layer.LSTM(256, return_sequences=True), merge_mode='concat', name="bidirectional_lSTM")(tensor)
        tensor = self.layer.Dense(self.class_num, name='dense2')(tensor)
        return tensor

    def ctc_lambda_func(self, args):
        y_pre, labels, input_len, label_len = args
        y_pre = y_pre[:, 2:, :]
        return self.backend.ctc_batch_cost(labels, y_pre, input_len, label_len)

    def create_model(self, data_format, training=True):
        if data_format == 'channels_first':
            # (batch, channels, height, width) default
            input_shape = (self.channels, self.image_height, self.image_width)
        else:
            # (batch, height, width, channels)
            assert data_format == 'channels_last'
            input_shape = (self.image_height, self.image_width, self.channels)
        inputs = self.layer.Input(shape=input_shape, name='input', dtype=tf.float32)
        # cnn
        tensor = self.build_cnn(inputs=inputs, data_format=data_format)
        # rnn
        tensor = self.build_rnn(tensor)
        # ctc
        y_pre = self.layer.Activation('softmax', name='softmax')(tensor)
        labels = self.layer.Input(name='labels', shape=[self.label_max_length], dtype=tf.float32)
        input_len = self.layer.Input(name='input_length', shape=[1], dtype=tf.int64)
        label_len = self.layer.Input(name='label_length', shape=[1], dtype=tf.int64)
        loss_out = self.layer.Lambda(self.ctc_lambda_func, output_shape=(1,), name='ctc')(
            [y_pre, labels, input_len, label_len])
        if training:
            return tf.keras.models.Model(inputs=[inputs, labels, input_len, label_len], outputs=loss_out)
        else:
            return tf.keras.models.Model(inputs=[inputs], outputs=y_pre)
