# ImageNet2012-tensorflow

#### 介绍
将ImageNet2012的前100类打包成tfrecord用于后续的模型训练，参考https://blog.csdn.net/gzroy/article/details/85954329

#### 使用说明

1.  使用torrent文件夹中的种子下载ILSVRC2012_img_train.tar和ILSVRC2012_img_val.tar，train文件137GB，val文件6.28GB，或者去官网下载。
2.  解压图片，代码可参考上面的blog或手动解压，我仅使用train中的前100类图片。
3.  使用create_tfrecord.py创建tfrecord，使用read_tfrecord.py读取
