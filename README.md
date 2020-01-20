# license_plate_recognize
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
* ![](https://github.com/MrZhousf/license_plate_recognize/blob/master/pic/data.png?raw=true)
* ![](https://github.com/MrZhousf/license_plate_recognize/blob/master/pic/data_detail.png?raw=true)

2. 保存车牌图片-提取图片中的车牌
```python3
fetch_plate_img(img_dir=img_dir_, save_dir=save_dir_)
```
运行完成后，只保留了车牌图片且命名为车牌号码
* ![](https://github.com/MrZhousf/license_plate_recognize/blob/master/pic/plate.png?raw=true)

3. 图片校验，删除有问题的图片
```python3
verify_img(img_dir=img_dir_, error_img_save_dir=error_img_save_dir_)
```

4. 统计出车牌中每个字符的个数
```python3
statistics(img_dir=img_dir_, log_txt=log_txt_)
```
统计结果如下，统计结果没有显示完全，可见车牌数据是安徽的居多（ccpd2019是中科大的学生收集与整理的）
* ![](https://github.com/MrZhousf/license_plate_recognize/blob/master/pic/statistic.png?raw=true)

5. 生成训练-评估数据：将数据按照百分比切割成训练集和评估集
```python3
generate_train_eval(img_dir=img_dir_, train_dir=train_dir_, eval_dir=eval_dir_, eval_percent=0.03)
```

### CNN+RNN
model目录下为网络训练业务
1. 神经网络：license_plate_model.py
2. 训练+评估：train.py
3. 数据生成器：data_generator.py
4. 预测/测试：prediction.py
* 下载预训练模型13_0.213.hdf5并置于train_dir目录下，该模型训练了13个epoch，loss=0.213：https://download.csdn.net/download/zsf442553199/12115514
* ![](https://github.com/MrZhousf/license_plate_recognize/blob/master/pic/pre.png?raw=true)

### 待完善
1. CCPD2019数据集有35万张车牌数据（包含各种天气），对于端到端的模型来说数据还有增加的空间
2. CCPD2019数据集未覆盖全国各地的车牌，安徽车牌居多，数据缺口较大
3. CCPD2019缺少新能源车、混动、货车以及特种车辆的车牌图片（黄牌、绿牌、黄绿牌、白牌、黑牌等）
4. 增加车牌检测网络，实现车牌检测+识别自动化，这个比较简单，可以采用yolov3实现，后续若有时间再提交一版

### 解决github上图片无法正常显示问题
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





