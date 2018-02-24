---
title: 论文 - Visualizing and Understanding ConvNet
date: 2018-02-08 10:48:21
tags:
    - paper
    - deep learning
---
[Visualizing & Understanding ConvNet](https://arxiv.org/pdf/1311.2901.pdf)这篇文章是比较早期关于CNN调试的文章，作者利用可视化方法，设计了一个超过AlexNet性能的网络结构。

![可视化结果](/img/paper_visconvnet_demo.png)
<!-- more -->

## 引言
继AlexNet之后，CNN在ImageNet竞赛中得到了广泛应用。AlextNet成功的原因包括以下三点：

- Large data。
- 硬件GPU性能。
- 一些技巧提升了模型的泛化能力，如Dropout技术。

不过CNN仍然像一只黑盒子，缺少可解释性。这使得对CNN的调试变得比较困难。我们提出了一种思路，可以找出究竟input中的什么东西对应了激活后的Feature map。

(对于神经网络的可解释性，可以从基础理论入手，也可以从实践中的经验入手。本文作者实际上就是在探索如何能够更好得使用经验对CNN进行调试。这种方法仍然没有触及到CNN本质的可解释性的东西，不过仍然在工程实践中有很大的意义，相当于将黑盒子变成了灰盒子。从人工取火到炼金术到现代化学，也不是这么一个过程吗？)

在AlexNet中，每个卷积单元常常由以下几个部分组成：

- 卷积层，使用一组学习到的$3D$滤波器与输入（上一层的输出或网络输入的数据）做卷积操作。
- 非线性激活层，通常使用`ReLU(x) = max(0, x)`。
- 可选的，池化层，缩小Feature map的尺寸。
- 可选的，LRN层（现在已经基本不使用）。

## DeconvNet
我们使用DeconvNet这项技术，寻找与输出的激活对应的输入模式。这样，我们可以看到，输入中的哪个部分被神经元捕获，产生了较强的激活。

对于一个CNN网络，我们考察某个激活Feature map的时候，将该层的其他Feature map弄成全$0$，然后逆着前向计算的过程做逆向操作。如图所示，展示了DeconvNet是如何构造的。
![DeconvNet的构造](/img/paer_visconvnet_deconvnet_structure.png)

