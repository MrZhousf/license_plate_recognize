# -*- coding:utf-8 -*-
# Author:      zhousf
# Date:        2019-07-26
# File:        train.py
# Description:  训练
from license_plate_workspace.model.model import Model
from license_plate_workspace.model.data_generator import ImageGenerator
import tensorflow as tf
import os


def ctc_loss_func():
    return {'ctc': lambda y_true, y_pred: y_pred}


def train(train_dir, train_img_dir, eval_img_dir, hdf5=None):
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    label_chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 "
    img_height = 64
    img_width = 128
    label_max_length = 9
    # data_format = 'channels_first'
    data_format = 'channels_last'
    train_generator = ImageGenerator(data_format=data_format, label_chars=label_chars, img_dir=train_img_dir,
                                     img_height=img_height, img_width=img_width, batch_size=128,
                                     label_max_length=label_max_length)
    eval_generator = ImageGenerator(data_format=data_format, label_chars=label_chars, img_dir=eval_img_dir,
                                    img_height=img_height, img_width=img_width, batch_size=16,
                                    label_max_length=label_max_length)
    checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=train_dir + '/{epoch:02d}_{val_loss:.3f}.h5',
                                                    monitor='loss', save_weights_only=True,
                                                    verbose=1, mode='min', period=1)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    model = Model(img_height,
                  img_width,
                  train_generator.class_num,
                  label_max_length).create_model(data_format=data_format)
    if hdf5 is not None:
        model.load_weights(hdf5)
    model.compile(optimizer=tf.keras.optimizers.Adadelta(), loss=ctc_loss_func())
    model.fit_generator(generator=train_generator,
                        steps_per_epoch=train_generator.steps_per_epoch,
                        epochs=1,
                        callbacks=[checkpoint],
                        validation_data=eval_generator,
                        use_multiprocessing=False,
                        validation_steps=eval_generator.steps_per_epoch)


if __name__ == '__main__':
    train_dir = "/media/ubuntu/b8f80802-d95a-41c3-b157-6f4e34967425/workspace/zhousf_projects/ml_project/license_plate_workspace/model/train_dir1"
    train_img_dir = "/media/ubuntu/b8f80802-d95a-41c3-b157-6f4e34967425/data-zhousf/plate_train"
    eval_img_dir = "/media/ubuntu/b8f80802-d95a-41c3-b157-6f4e34967425/data-zhousf/plate_eval"
    # train_img_dir = "/media/ubuntu/b8f80802-d95a-41c3-b157-6f4e34967425/data-zhousf/test/train"
    # eval_img_dir = "/media/ubuntu/b8f80802-d95a-41c3-b157-6f4e34967425/data-zhousf/test/eval"
    train(train_dir, train_img_dir, eval_img_dir)
    pass
