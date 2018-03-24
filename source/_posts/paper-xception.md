---
title: 论文 - Xception, Deep Learning with Depthwise separable Convolution
date: 2018-03-22 09:44:38
tags:
    - paper
    - deep learning
---
在MobileNet, ShuffleMet等轻量级网络中，**depthwise separable conv**是一个很流行的设计。借助[Xception: Deep Learning with Depthwise separable Convolution](https://arxiv.org/abs/1610.02357)，对这种分解卷积的思路做一个总结。
<!-- more -->

## 起源
自从AlexNet以来，DNN的网络设计经过了ZFNet->VGGNet->GoogLeNet->ResNet等几个发展阶段。本文作者的思路正是受GoogLeNet中Inception结构启发。Inception结构是最早有别于VGG等“直筒型”结构的网络module。以Inception V3为例，一个典型的Inception模块长下面这个样子：
![一个典型的Inception结构](/img/paper-xception-inception-module.png)

对于一个CONV层来说，它要学习的是一个$3D$的filter，包括两个空间维度（spatial dimension），即width和height；以及一个channel dimension。这个filter和输入在$3$个维度上进行卷积操作，得到最终的输出。可以用伪代码表示如下：
```
// 对于第i个filter
// 计算输入中心点(x, y)对应的卷积结果
sum = 0
for c in 1:C
  for h in 1:K
    for w in 1:K
      sum += in[c, y-K/2+h, x-K/2+w] * filter_i[c, h, w]
out[i, y, x] = sum
```

可以看到，在$3D$卷积中，channel这个维度和spatial的两个维度并无不同。

在Inception中，卷积操作更加轻量级。输入首先被$1\times 1$的卷积核处理，得到了跨channel的组合(cross-channel correlation)，同时将输入的channel dimension减少了$3\sim 4$倍（一会$4$个支路要做`concat`操作）。这个结果被后续的$3\times 3$卷积和$5\times 5$卷积核处理，处理方法和普通的卷积一样，见上。

由此作者想到，Inception能够work证明后面的一条假设就是：卷积的channel相关性和spatial相关性是可以解耦的，我们没必要要把它们一起完成。

## 简化Inception，提取主要矛盾
接着，为了更好地分析问题，作者将Inception结构做了简化，保留了主要结构，去掉了AVE Pooling操作，如下所示。
![简化后的Inception](/img/paper-xception-simplified-inception-module.png)

好的，我们现在将底层的$3$个$1\times 1$的卷积核组合起来，其实上面的图和下图是等价的。一个“大的”$1\times 1$的卷积核（channels数目变多），它的输出结果在channel上被分为若干组（group），每组分别和不同的$3\times 3$卷积核做卷积，再将这$3$份输出拼接起来，得到最后的输出。
![另一种形式](/img/paper-xception-equivalent-inception-module.png)

那么，如果我们把分组数目继续调大呢？极限情况，我们可以使得group number = channel number，如下所示：
![极限模式](/img/paper-xception-extreme-version.png)


## Depthwise Separable Conv
这种结构和一种名为**depthwise separable conv**的技术很相似，即首先使用group conv在spatial dimension上卷积，然后使用$1\times 1$的卷积核做cross channel的卷积（又叫做*pointwise conv*）。主要有两点不同：

- 操作的顺序。在TensorFlow等框架中，depthwise separable conv的实现是先使用channelwise的filter只在spatial dimension上做卷积，再使用$1\times 1$的卷积核做跨channel的融合。而Inception中先使用$1\times 1$的卷积核。
- 非线性变换的缺席。在Inception中，每个conv操作后面都有ReLU的非线性变换，而depthwise separable conv没有。

第一点不同不太重要，尤其是在深层网络中，这些block都是堆叠在一起的。第二点论文后面通过实验进行了比较。可以看出，去掉中间的非线性激活，能够取得更好的结果。
![非线性激活的影响](/img/paper-xception-experiment-intermediate-activation.png)

## Xception网络架构
基于上面的分析，作者认为这样的假设是合理的：cross channel的相关和spatial的相关可以**完全**解耦。

> we make the following hypothesis: that the mapping of cross-channels correlations and spatial correlations in the feature maps of convolutional neural networks can be *entirely* decoupled. 

Xception的结构基于ResNet，但是将其中的卷积层换成了depthwise separable conv。如下图所示。整个网络被分为了三个部分：Entry，Middle和Exit。

> The Xception architecture: the data first goes through the entry flow, then through the middle flow which is repeated eight times, and finally through the exit flow. Note that all Convolution and SeparableConvolution layers are followed by batch normalization [7] (not included in the diagram). All SeparableConvolution layers use a depth multiplier of 1 (no depth expansion).

![Xception的网络结构](/img/paper-xception-arch.png)
