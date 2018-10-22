---
title: 论文 - Rethinking The Value of Network Pruning 
date: 2018-10-22 22:25:42
tags:
    - deep learning
    - model compression
    - paper
---
[这篇文章](https://openreview.net/forum?id=rJlnB3C5Ym)是ICLR 2019的投稿文章，最近也引发了大家的注意。在我的博客中，已经对此做过简单的介绍，请参考[论文总结 - 模型剪枝 Model Pruning](https://xmfbit.github.io/2018/10/03/paper-summary-model-pruning/)。

这篇文章的主要观点在于想纠正人们之前的认识误区。当然这个认识误区和DL的发展是密不可分的。DL中最先提出的AlexNet是一个很大的模型。后面的研究者虽然也在不断发明新的网络结构（如inception，Global Pooling，ResNet等）来获得参数更少更强大的模型，但模型的size总还是很大。既然研究社区是从这样的“大”模型出发的，那当面对工程上需要小模型以便在手机等移动设备上使用时，很自然的一条路就是去除大模型中已有的参数从而得到小模型。也是很自然的，我们需要保留大模型中“有用的”那些参数，让小模型以此为基础进行fine tune，补偿因为去除参数而导致的模型性能下降。

然而，自然的想法就是合理的么？这篇文章对此提出了质疑。这篇论文的主要思路已经在上面贴出的博文链接中说过了。这篇文章主要是结合作者开源的代码对论文进行梳理：[Eric-mingjie/rethinking-network-pruning](https://github.com/Eric-mingjie/rethinking-network-pruning)。

<!-- more -->

## FLOP的计算
代码中有关于PyTorch模型的FLOPs的计算，见[compute_flops.py](https://github.com/Eric-mingjie/rethinking-network-pruning/blob/master/imagenet/l1-norm-pruning/compute_flops.py)。可以很方便地应用到自己的代码中。

## ThiNet的实现

## 实验比较

## 结论
几个仍然有疑问的地方：

1. 作者已经证明在ImageNet/CIFAR等样本分布均衡的数据集上的结论，如果样本分布不均衡呢？有三种思路有待验证：
  - prune模型需要从大模型处继承权重，然后直接在不均衡数据集上训练即可；
  - prune模型不需要从大模型处继承权重， 但是需要先在ImageNet数据集上训练，然后再在不均衡数据集上训练；
  - prune模型直接在不均衡数据集上训练（以我的经验，这种思路应该是不work的）

2. prune前的大模型权重不重要，结构重要，这是本文的结论之一。自动搜索树的prune算法可以看做是模型结构搜索，但是大模型给出了搜索空间的一个很好的初始点。这个初始点是否是任务无关的？也就是说，对A任务有效的小模型，是否在B任务上也是很work的？

3. 现在的网络搜索中应用了强化学习/遗传算法等方法，这些方法怎么能够和prune结合？ECCV 2018中HanSong和He Yihui发表了AMC方法。

总之，作者用自己辛勤的实验，给我们指出了一个"可能的"（毕竟文章还没被接收）误区，但是仍然有很多乌云漂浮在上面，需要更多的实验。
