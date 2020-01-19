# -*- coding:utf-8 -*-
# Author:      zhousf
# Date:        2019-09-11
# File:        deal_ccpd_data.py
# Description:  CCPD2019
# https://github.com/detectRecog/CCPD
import os
import cv2
from core.util import img_util
import random
import math
from core.util import file_util

provinces = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂",
             "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "警", "学", "O"]
provinces_code = ["00", "01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12", "13", "14", "15", "16",
                  "17", "18", "19", "20",
                  "21", "22", "23", "24", "25", "26", "27", "28", "29", "30", "31", "32", "33"]
alphabets = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W',
             'X', 'Y', 'Z', 'O']
ads = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X',
       'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'O']


def fetch_plate(img):
    plate_str = ''
    inf = img.split('&')
    if len(inf) > 2:
        img_name = inf[-1]
        plate_ = img_name.split('-')[1]
        numbers = plate_.split('_')
        if len(numbers) < 7:
            return ''
        plate_str = provinces_code[int(numbers[0])]
        plate_str += alphabets[int(numbers[1])]
        plate_str += ads[int(numbers[2])]
        plate_str += ads[int(numbers[3])]
        plate_str += ads[int(numbers[4])]
        plate_str += ads[int(numbers[5])]
        plate_str += ads[int(numbers[6])]
    return plate_str


def fetch_box(img):
    box_ = []
    inf = img.split('&')
    if len(inf) > 2:
        img = img.rsplit('/', 1)[-1].rsplit('.', 1)[0].split('-')
        left_up, right_bottom = [[int(eel) for eel in el.split('&')] for el in img[2].split('_')]
        box_.append((int(left_up[0] - 5), int(left_up[1])))
        box_.append((int(right_bottom[0]), int(right_bottom[1])))
    return box_


def fetch_plate_img(img_dir, save_dir):
    """
    保存车牌图片
    :param img_dir:
    :param save_dir:
    :return:
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for root, dirs, files in os.walk(img_dir):
        for img in files:
            if not img_util.is_img(img):
                continue
            plate = fetch_plate(img)
            box = fetch_box(img)
            if len(box) == 0:
                continue
            print(plate)
            image = cv2.imread(os.path.join(root, img))
            # 开始的y坐标:结束的y坐标,开始x:结束的x
            crop_img = image[int(box[0][1]):int(box[1][1]), int(box[0][0]): int(box[1][0])]
            # plate_box = cv2.rectangle(image, box[0], box[1], (0, 255, 0), 2)
            ran = random.randint(1000, 9999)
            cv2.imwrite(os.path.join(save_dir, '{0}_{1}.jpg'.format(plate, ran)), crop_img)


def statistics(img_dir, log_txt):
    """
    统计出车牌中每个字符的个数
    :param img_dir:
    :param log_txt:
    :return:
    """
    letters = {}
    plates = {}
    total = 0
    tmp = {p for p in provinces}
    tmp.update({p for p in alphabets})
    tmp.update({p for p in ads})
    for char in tmp:
        letters[char] = 0
    for root, dirs, files in os.walk(img_dir):
        for img in files:
            if not img_util.is_img(img):
                continue
            plate, ext = img.split('_')
            if plate in plates:
                plates[plate] += 1
            else:
                plates[plate] = 1
            total += 1
            print(plate)
            plate = list(plate)
            for char in plate:
                if char in letters:
                    letters[char] += 1
    with open(log_txt, "w+") as f:
        sta = "车牌图片{0}张，车牌{1}个(去重)\n".format(total, len(plates))
        f.write(sta)
        for char in letters:
            le = "{0}:{1}".format(char, letters.get(char))
            print(le)
            f.write(le + '\n')
        print(sta)


def generate_train_eval(img_dir, train_dir, eval_dir, eval_percent=0.05):
    """
    生成训练-评估数据
    :param img_dir:
    :param train_dir:
    :param eval_dir:
    :param eval_percent: 评估数据占比%
    :return:
    """
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    if not os.path.exists(eval_dir):
        os.makedirs(eval_dir)
    train_num = 0
    eval_num = 0
    for root, dirs, files in os.walk(img_dir):
        for d in dirs:
            dir_path = os.path.join(root, d)
            img_num = len(os.listdir(dir_path))
            val_num = math.ceil(img_num * eval_percent)
            images = os.listdir(dir_path)
            for img in images:
                if not img_util.is_img(img):
                    continue
                print(os.path.join(dir_path, img))
                if val_num > 0:
                    file_util.copy_file(os.path.join(dir_path, img), eval_dir)
                    val_num -= 1
                    eval_num += 1
                    print(os.path.join(dir_path, img))
                else:
                    train_num += 1
                    file_util.copy_file(os.path.join(dir_path, img), train_dir)
    print("train: {0}张，eval: {1}张".format(train_num, eval_num))


def rename_plate(img_dir):
    """
    将车牌中文转成英文
    :param img_dir:
    :return:
    """
    for root, dirs, files in os.walk(img_dir):
        for img in files:
            if not img_util.is_img(img):
                continue
            index = provinces.index(img[0])
            pro = provinces_code[index]
            new_img = img.replace(img[0], pro)
            os.rename(os.path.join(root, img), os.path.join(root, new_img))


def verify_img(img_dir, error_img_save_dir):
    """
    图片校验，删除有问题的图片
    :param img_dir:
    :param error_img_save_dir:
    :return:
    """
    total = 0
    remove = 0
    if not os.path.exists(error_img_save_dir):
        os.makedirs(error_img_save_dir)
    for root, dirs, files in os.walk(img_dir):
        for img in files:
            if not img_util.is_img(img):
                continue
            total += 1
            img_file = os.path.join(root, img)
            image = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)
            if image is None:
                print(img_file)
                file_util.move_file(img_file, error_img_save_dir)
                remove += 1
    print("共{0}张图片，有问题的图片{1}张".format(total, remove))


if __name__ == "__main__":
    img_dir_ = '/media/ubuntu/b8f80802-d95a-41c3-b157-6f4e34967425/data-zhousf/plate_train'
    save_dir_ = '/media/ubuntu/b8f80802-d95a-41c3-b157-6f4e34967425/data-zhousf/plate/base'
    log_txt_ = '/media/ubuntu/b8f80802-d95a-41c3-b157-6f4e34967425/data-zhousf/plate/readme.txt'

    # step 1 : 保存车牌图片-提取图片中的车牌
    # fetch_plate_img(img_dir=img_dir_, save_dir=save_dir_)

    # step 2 : 图片校验，删除有问题的图片
    # error_img_save_dir_ = "/media/ubuntu/b8f80802-d95a-41c3-b157-6f4e34967425/data-zhousf/error_train"
    # verify_img(img_dir=img_dir_, error_img_save_dir=error_img_save_dir_)

    # step 3 : 统计出车牌中每个字符的个数
    # statistics(img_dir=img_dir_, log_txt=log_txt_)
    # train_dir_ = '/media/ubuntu/b8f80802-d95a-41c3-b157-6f4e34967425/data-zhousf/plate_train'
    # eval_dir_ = '/media/ubuntu/b8f80802-d95a-41c3-b157-6f4e34967425/data-zhousf/plate_eval'
    
    # step 4 : 生成训练-评估数据
    # generate_train_eval(img_dir=img_dir_, train_dir=train_dir_, eval_dir=eval_dir_, eval_percent=0.03)

