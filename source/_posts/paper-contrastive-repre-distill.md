---
layout: post
title: 论文阅读 - Contrastive Representation Distillation
date: 2020-05-05 16:45:12
tags:
    - paper
    - model compression
    - contrastive learning
---


# 预备知识

补充一些关于信息论的预备知识。

## [信息量](https://en.wikipedia.org/wiki/Information_content)

随机变量$X$取值$x$的信息量定义为：

$$I_X(x) = -\log P_X(x)$$

要想形式化地理解为何信息量是如上所示的对数函数，可以从下面几个公理（axiom）出发：

- An event with probability 100% is perfectly unsurprising and yields no information.
- The less probable an event is, the more surprising it is and the more information it yields.
- If two independent events are measured separately, the total amount of information is the sum of the self-informations of the individual events.

其中，最后一条意味着$f(xy) = f(x)f(y)$，而前面两条得到常数项和单调递减。

## [信息熵](https://en.wikipedia.org/wiki/Entropy_(information_theory))

由上可知，随机变量的信息量也是一个随机变量。随机变量$X$的信息熵，又叫做自信息（self information)，是其信息量的期望。这里假定其为离散随机变量，将期望符号展开：

$$H(X) = E[I(X)] = E[-\log P(X)] = -\sum P(x)\log P(x)$$

## [互信息](https://en.wikipedia.org/wiki/Mutual_information)

两个随机变量$X$和$Y$，它们的互信息（mutual information）是它们联合概率密度函数和边缘概率密度函数的乘积的KL散度：

$$MI(X;Y) = D_{KL}(P_{X,Y}||P_XP_Y) = E_{P_{X,Y}}[\log\frac{P_{X, Y}}{P_X P_Y}] = \sum_X\sum_Y P_{X,Y}\log\frac{P_{X,Y}}{P_X P_Y}$$