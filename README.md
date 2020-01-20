# license_plate_recongnize
端到端车牌识别项目，完整的数据集、数据制作、训练、评估、预测业务
* 运行平台：tensorflow1.12.0+python3.6
* 神经网络：CNN+RNN
* 数据集：CCPD2019

### 依赖
pip install -r requirements.txt

### 数据集
CCPD2019车牌数据集：https://github.com/detectRecog/CCPD

### 训练数据生成
生成训练数据集： deal_ccpd_data.py
1. 下载CCPD2019数据集：https://github.com/detectRecog/CCPD
* 共有9种类型的车牌数据
* ![](https://github.com/MrZhousf/license_plate_recongnize/blob/master/pic/data.png?raw=true)
* ![](https://github.com/MrZhousf/license_plate_recongnize/blob/master/pic/data_detail.png?raw=true)

2. 保存车牌图片-提取图片中的车牌
```python3
fetch_plate_img(img_dir=img_dir_, save_dir=save_dir_)
```
运行完成后，只保留了车牌图片且命名为车牌号码
* ![](https://github.com/MrZhousf/license_plate_recongnize/blob/master/pic/plate.png?raw=true)

3. 图片校验，删除有问题的图片
```python3
verify_img(img_dir=img_dir_, error_img_save_dir=error_img_save_dir_)
```

4. 统计出车牌中每个字符的个数
```python3
statistics(img_dir=img_dir_, log_txt=log_txt_)
```
统计结果如下，统计结果没有显示完全，可见车牌数据是安徽的居多（ccpd2019是中科大的学生收集与整理）
* ![](https://github.com/MrZhousf/license_plate_recongnize/blob/master/pic/statistic.png?raw=true)

5. 生成训练-评估数据：将数据按照百分比切割成训练集和评估集
```python3
generate_train_eval(img_dir=img_dir_, train_dir=train_dir_, eval_dir=eval_dir_, eval_percent=0.03)
```

### CNN+RNN
model目录下为网络训练业务
1. 神经网络：model.py
2. 训练+评估：train.py
3. 数据生成器：data_generator.py
4. 预测/测试：prediction.py
* ![](https://github.com/MrZhousf/license_plate_recongnize/blob/master/pic/pre.png?raw=true)

### github上图片无法正常显示
* 在终端执行
```python3
sudo vi /etc/hosts
```
* 添加以下地址
```python3
# GitHub Start
192.30.253.112    github.com
192.30.253.119    gist.github.com
199.232.28.133    assets-cdn.github.com
199.232.28.133    raw.githubusercontent.com
199.232.28.133    gist.githubusercontent.com
199.232.28.133    cloud.githubusercontent.com
199.232.28.133    camo.githubusercontent.com
199.232.28.133    avatars0.githubusercontent.com
199.232.28.133    avatars1.githubusercontent.com
199.232.28.133    avatars2.githubusercontent.com
199.232.28.133    avatars3.githubusercontent.com
199.232.28.133    avatars4.githubusercontent.com
199.232.28.133    avatars5.githubusercontent.com
199.232.28.133    avatars6.githubusercontent.com
199.232.28.133    avatars7.githubusercontent.com
199.232.28.133    avatars8.githubusercontent.com
 # GitHub End
```
* 刷新页面即可





