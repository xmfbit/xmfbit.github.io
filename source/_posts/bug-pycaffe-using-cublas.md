---
title: 捉bug记 - Cannot create Cublas handle. Cublas won't be available.
date: 2018-02-08 14:16:53
tags:
    - caffe
    - python
    - debug
---
这两天在使用Caffe的时候出现了一个奇怪的bug。当使用C++接口时，完全没有问题；但是当使用python接口时，会出现错误提示如下：
```
common.cpp:114] Cannot create Cublas handle. Cublas won't be available.
common.cpp:121] Cannot create Curand generator. Curand won't be available.
```
<!-- more -->

令人疑惑的是，这个python脚本我前段时间已经用过几次了，却没有这样的问题。

如果在Google上搜索这个问题，很多讨论都是把锅推给了驱动，不过我使用的这台服务器并没有更新过驱动或系统。本来想要试试重启大法，但是上面还有其他人在跑的任务，所以重启不太现实。

最后找到了这个issue: [Cannot use Caffe on another GPU when GPU 0 has full memory](https://github.com/BVLC/caffe/issues/440)。联想到我目前使用的服务器上GPU０也正是在跑着一项很吃显存的任务（如下所示），所以赶紧试了一下里面@longjon的方法。
![nvidia-smi给出的显卡使用信息](/img/bug_pycaffe_nvidia_smi_result.png)

使用`CUDA_VISIBLE_DEVICES`变量，指定Caffe能看到的显卡设备。
```　bash
CUDA_VISIBLE_DEVICES=2 python my_script.py --gpu_id=0
```

果然就可以了！

这个问题应该出在pycaffe的初始化上。这里不再深究。