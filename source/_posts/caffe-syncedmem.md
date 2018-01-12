---
title: Caffe 中的 SyncedMem介绍
date: 2018-01-12 14:05:59
tags:
     - caffe
---
`Blob`是Caffe中的基本数据结构，类似于TensorFlow和PyTorch中的Tensor。图像读入后作为`Blob`，开始在各个`Layer`之间传递，最终得到输出。下面这张图展示了`Blob`和`Layer`之间的关系：
 <img src="/img/caffe_syncedmem_blob_flow.jpg" width = "300" height = "200" alt="blob的流动" align=center />

Caffe中的`Blob`在实现的时候，使用了`SyncedMem`管理内存，并在内存（Host）和显存（device）之间同步。这篇博客对Caffe中`SyncedMem`的实现做一总结。
<!-- more -->

## SyncedMem的作用
`Blob`是一个多维的数组，可以位于内存，也可以位于显存（当使用GPU时）。一方面，我们需要对底层的内存进行管理，包括何何时开辟内存空间。另一方面，我们的训练数据常常是首先由硬盘读取到内存中，而训练又经常使用GPU，最终结果的保存或可视化又要求数据重新传回内存，所以涉及到Host和Device内存的同步问题。

## 同步的实现思路
在`SyncedMem`的实现代码中，作者使用一个枚举量`head_`来标记当前的状态。如下所示：

``` cpp
// in SyncedMem
enum SyncedHead { UNINITIALIZED, HEAD_AT_CPU, HEAD_AT_GPU, SYNCED };
SuncedHead head_;
```

这样，利用`head_`变量，就可以构建一个状态转移图，在不同状态下进行必要的同步操作等。
![状态转换图](/img/caffe_syncedmem_transfer.png)

## 具体实现