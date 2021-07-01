# README

### <u>运行环境</u>

##### tensorflow 1.4.1+  <2.0

##### opencv3

##### slidingwindow

##### coco dataset

##### swig

下载swig：[https://sourceforge.net/projects/swig/](https://sourceforge.net/projects/swig/)

将swig.exe放到系统环境变量path下面

在cmd中运行

```
cd tf_pose/pafprocess
swig -python -c++ pafprocess.i && python3 setup.py build_ext --inplace
```

##### 依赖包

```
argparse
dill
fire
matplotlib
numba
psutil
pycocotools
requests
opencv-python
scikit-image
scipy
numpy
slidingwindow
tensorflow
tqdm
```



### <u>测试</u>

以视频测试为例：

运行 run_video.py，在44行和52行中修改视频名称，26行中add_argument的resize参数修改256*256速度会

变快，model参数可设置为想要的模型，如果不想要视频背景，解开61行和62行的注释并将draw_humans中

的image改为emptyImage，可得到openpose效果



若想观测我们的改进效果

运行 ./tfpose/estimator.py

运行 zunfantry.py

运行 test.py，5行和8行可修改视频名称，得到两个效果视频



### <u>我们的改进做了什么</u>

##### 数据增强

对数据集进行了缩放，旋转，翻转，裁剪。

##### 测试视频自身叠加效果

将一个测试用的视频截取成大致相同但有时间差的两个视频，时间差在0-0.5s之间，现在的结果是越小越好，

记为视频00 和视频01

视频00训练时记录下其产生humans数据，记为prehumans，将prehumans传入视频01的训练中，如果视

频01产生的humans数据的关键点序列中，某个位置没有数据而prehumans的关键点序列中存在数据，则将

prehumans该位置的数据存入humans中

具体实现在 ./tfpose/estimator.py













​		





