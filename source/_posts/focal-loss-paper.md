---
title: Focal Loss论文阅读 - Focal Loss for Dense Object Detection
date: 2017-08-14 22:43:55
tags:
    - paper
    - deep learning
---
Focal Loss这篇文章是He Kaiming和Ross发表在ICCV2017上的文章。关于这篇文章在知乎上有相关的[讨论](https://www.zhihu.com/question/63581984)。最近一直在做强化学习相关的东西，目标检测方面很长时间不看新的东西了，把自己阅读论文的要点记录如下，也是一次对这方面进展的回顾。

下图来自于论文，是各种主流模型的比较。其中横轴是前向推断的时间，纵轴是检测器的精度。作者提出的RetinaNet在单独某个维度上都可以吊打其他模型。不过图上没有加入YOLO的对比。YOLO的速度仍然是其一大优势，但是精度和其他方法相比，仍然不高。

![不同模型关于精度和速度的比较](/img/focal_loss_different_model_comparison.jpg)
<!-- more -->

## 为什么要有Focal Loss？
目前主流的检测算法可以分为两类：one-state和two-stage。前者以YOLO和SSD为代表，后者以RCNN系列为代表。后者的特点是分类器是在一个稀疏的候选目标中进行分类（背景和对应类别），而这是通过前面的proposal过程实现的。例如Seletive Search或者RPN。相对于后者，这种方法是在一个稀疏集合内做分类。与之相反，前者是输出一个稠密的proposal，然后丢进分类器中，直接进行类别分类。后者使用的方法结构一般较为简单，速度较快，但是目前存在的问题是精度不高，普遍不如前者的方法。

论文作者指出，之所以做稠密分类的后者精度不够高，核心问题（central issus）是稠密proposal中前景和背景的极度不平衡。以我更熟悉的YOLO举例子，比如在PASCAL VOC数据集中，每张图片上标注的目标可能也就几个。但是YOLO V2最后一层的输出是$13 \times 13 \times 5$，也就是$845$个候选目标！大量（简单易区分）的负样本在loss中占据了很大比重，使得有用的loss不能回传回来。

基于此，作者将经典的交叉熵损失做了变形（见下），给那些易于被分类的简单例子小的权重，给不易区分的难例更大的权重。同时，作者提出了一个新的one-stage的检测器RetinaNet，达到了速度和精度很好地trade-off。

$$\text{FL}(p_t) = -(1-p_t)^\gamma \log(p_t)$$

## 物体检测的两种主流方法
在深度学习之前，经典的物体检测方法为滑动窗，并使用人工设计的特征。HoG和DPM等方法是其中比较有名的。

R-CNN系的方法是目前最为流行的物体检测方法之一，同时也是目前精度最高的方法。在R-CNN系方法中，正负类别不平衡这个问题通过前面的proposal解决了。通过EdgeBoxes，Selective Search，DeepMask，RPN等方法，过滤掉了大多数的背景，实际传给后续网络的proposal的数量是比较少的（1-2K）。

在YOLO，SSD等方法中，需要直接对feature map的大量proposal（100K）进行检测，而且这些proposal很多都在feature map上重叠。大量的负样本带来两个问题：
- 过于简单，有效信息过少，使得训练效率低；
- 简单的负样本在训练过程中压倒性优势，使得模型发生退化。

在Faster-RCNN方法中，Huber Loss被用来降低outlier的影响（较大error的样本，也就是难例，传回来的梯度做了clipping，也只能是$1$）。而FocalLoss是对inner中简单的那些样本对loss的贡献进行限制。即使这些简单样本数量很多，也不让它们在训练中占到优势。

## Focal Loss
Focal Loss从交叉熵损失而来。二分类的交叉熵损失如下：
$$\text{CE}(p, y) = \begin{cases}-\log(p) \quad &\text{if}\quad y = 1\\ -\log(1-p) &\text{otherwise}\end{cases}$$
