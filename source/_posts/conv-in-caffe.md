---
title: Caffe中卷积的大致实现思路
date: 2018-02-26 15:26:09
tags:
     - caffe
     - deep learning
---
参考资料：知乎：[在Caffe中如何计算卷积](https://www.zhihu.com/question/28385679)。
![Naive Loop](/img/conv-in-caffe-naive-loop.png)
<!-- more -->

使用`im2col`将输入的图像或特征图转换为矩阵，后续就可以使用现成的线代运算优化库，如BLAS中的GEMM，来快速计算。
![im2col->gemm](/img/conv-in-caffe-im2col-followed-gemm.png)

im2col的工作原理如下：每个要和卷积核做卷积的patch被抻成了一个feature vector。不同位置的patch，顺序堆叠起来，
![patches堆起来](/img/conv-in-caffe-im2col-1.png)

最后就变成了这样：
![最后的样子](/img/conv-in-caffe-im2col-2.png)

同样的，对卷积核也做类似的变换。将单一的卷积核抻成一个行向量，然后把`c_out`个卷积核顺序排列起来。
![卷积核 to col](/img/conv-in-caffe-im2col-3.png)

我们记图像那个矩阵是`A`，记卷积那个矩阵是`F`。那么，对于第`i`个卷积核来说，它现在实际上是`F`里面的第`i`个行向量。为了计算它在原来图像上的各个位置的卷积，现在我们需要它和矩阵`A`中的每行做点积。也就是 `F_i * [A_1^T, A_2^T, … A_i^T]` （也就是`A`的转置）。推广到其他的卷积核，就是说，最后的结果是`F*A^T`.

我们可以用矩阵维度验证。`F`的维度是`Cout x (C x K x K)`. 输入的Feature map matrix的维度是`(H x W) x (C x K x K)`。那么上述矩阵乘法的结果就是 `Cout x (H x W)`。正好可以看做输出的三维blob的大小：`Cout x H x W`。

这里[Convolution in Caffe: a memo](https://github.com/Yangqing/caffe/wiki/Convolution-in-Caffe:-a-memo)还有贾扬清对于自己当时在caffe中实现conv的”心路历程“，题图出自此处。