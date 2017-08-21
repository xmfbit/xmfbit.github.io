---
title: Focal Loss论文阅读 - Focal Loss for Dense Object Detection
date: 2017-08-14 22:43:55
tags:
    - paper
    - deep learning
---
Focal Loss这篇文章是He Kaiming和Ross发表在ICCV2017上的文章。关于这篇文章在知乎上有相关的[讨论](https://www.zhihu.com/question/63581984)。最近一直在做强化学习相关的东西，目标检测方面很长时间不看新的东西了，把自己阅读论文的要点记录如下，也是一次对这方面进展的回顾。

下图来自于论文，是各种主流模型的比较。其中横轴是前向推断的时间，纵轴是检测器的精度。作者提出的RetinaNet在单独某个维度上都可以吊打其他模型。不过图上没有加入YOLO的对比。YOLO的速度仍然是其一大优势。

![不同模型关于精度和速度的比较](/img/focal_loss_different_model_comparison.jpg)
<!-- more -->

## 为什么要有Focal Loss？
目前主流的检测算法可以分为两类：one-state和two-stage。前者以YOLO和SSD为代表，后者以RCNN系列为代表。后者的特点是分类器是在一个稀疏的候选目标中进行分类（背景和对应类别），而这是通过前面的proposal过程实现的。例如Seletive Search或者RPN。与之相反，前者是输出一个稠密的proposal，
