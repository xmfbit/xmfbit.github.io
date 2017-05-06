---
title: Neural Network for Machine Learning - Lecture 01
date: 2017-05-03 20:56:36
tags:
    - deep learning
    - 公开课
---
Hinton 在 Coursera 课程“Neural Network for Machine Learning”新开了一个班次。对课程内容做一总结。课程内容也许已经跟不上最近DL的发展，不过还是有很多的好东西。
![神经元](/img/hinton_brainsimulator.jpg)
<!-- more -->

## Why do we need ML?
从数据中学习，无需手写大量的逻辑代码。ML算法从数据中学习，给出完成任务（如人脸识别）的程序。这里的程序可能有大量的参数组成。得益于硬件的发展和大数据时代的来临，机器学习的应用越来越广泛。如下图所示，MNIST数据集中的手写数字2变化多端，很难人工设计代码逻辑找到判别一个手写数字是不是2的方法。
![MNIST Digit 2](/img/hinton_01_mnist_example.png)

## What are neural network?
为什么要研究神经网络？

- 理解人脑机理的一个途径；
- 受到神经系统启发的并行计算
- 新的学习算法来解决现实问题（本课所关心的）

（这里总结的不是很科学，勉强概括了讲义的内容）
神经元结构如下所示。树突（dendritic tree）和其他神经元相连作为输入，轴突（axon）发散出很多分支和其他神经元相连。轴突和树突之间通过突触（synapse）连接。轴突有足够的电荷产生兴奋。这样完成神经元到神经元的communication。
![神经元的结构](/img/hinton_01_neuron_structure.png)

神经元之间互相连接。对不同的神经元输入，有不同的权重。这些权重可以变化，使得神经元之间的连接或变得更加紧密或疏离。人类大脑的神经元多达$10^{11}$个，每一个都有多达$10^4$个连接权重。不同神经元分布式计算，带宽很大。
![神经元的相互连接](/img/hinton_01_neuron_commucation.png)

大脑中不同的神经元分工不同（局部损坏造成相应的身体功能受损），但是这些神经元长得都差不多，它们也可以在一定的环境下发育成特定功能的神经元。

而人工神经网络就是根据神经元的兴奋传导机理，人工模拟的神经网络。

## Different neurons
