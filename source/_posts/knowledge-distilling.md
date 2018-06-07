---
title: 论文 - Distilling the Knowledge in a Neural Network
date: 2018-06-07 21:56:12
tags:
---
知识蒸馏（Knowledge Distilling）是模型压缩的一种方法，是指利用已经训练的一个较复杂的Teacher模型，指导一个较轻量的Student模型训练，从而在减小模型大小和计算资源的同时，尽量保持原Teacher模型的准确率的方法。这种方法受到大家的注意，主要是由于Hinton的论文[Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531)。这篇博客做一总结。后续还会有KD方法的改进相关论文的心得介绍。

<!--more -->

## 背景
这里我将Wang Naiyang在知乎相关问题的[回答](https://www.zhihu.com/question/50519680/answer/136363665
)粘贴如下，将KD方法的motivation讲的很清楚。图森也发了论文对KD进行了改进，下篇笔记总结。

> Knowledge Distill是一种简单弥补分类问题监督信号不足的办法。传统的分类问题，模型的目标是将输入的特征映射到输出空间的一个点上，例如在著名的Imagenet比赛中，就是要将所有可能的输入图片映射到输出空间的1000个点上。这么做的话这1000个点中的每一个点是一个one hot编码的类别信息。这样一个label能提供的监督信息只有log(class)这么多bit。然而在KD中，我们可以使用teacher model对于每个样本输出一个连续的label分布，这样可以利用的监督信息就远比one hot的多了。另外一个角度的理解，大家可以想象如果只有label这样的一个目标的话，那么这个模型的目标就是把训练样本中每一类的样本强制映射到同一个点上，这样其实对于训练很有帮助的类内variance和类间distance就损失掉了。然而使用teacher model的输出可以恢复出这方面的信息。具体的举例就像是paper中讲的， 猫和狗的距离比猫和桌子要近，同时如果一个动物确实长得像猫又像狗，那么它是可以给两类都提供监督。综上所述，KD的核心思想在于"打散"原来压缩到了一个点的监督信息，让student模型的输出尽量match teacher模型的输出分布。其实要达到这个目标其实不一定使用teacher model，在数据标注或者采集的时候本身保留的不确定信息也可以帮助模型的训练。

## 蒸馏

这篇论文很好阅读。论文中实现蒸馏是靠soften softmax prob实现的。在分类任务中，常常使用交叉熵作为损失函数，使用one-hot编码的标注好的类别标签$\{1,2,\dots,K\}$作为target，如下所示：
$$\mathcal{L} = -\sum_{i=1}^{K}t_i\log p_i$$

作者指出，粗暴地使用one-hot编码丢失了类间和类内关于相似性的额外信息。举个例子，在手写数字识别时，$2$和$3$就长得很像。但是使用上述方法，完全没有考虑到这种相似性。对于已经训练好的模型，当识别数字$2$时，很有可能它给出的概率是：数字$2$为$0.99$，数字$3$为$10^{-2}$，数字$7$为$10^{-4}$。如何能够利用训练好的Teacher模型给出的这种信息呢？

可以使用带温度的softmax函数。对于softmax的输入（下文统一称为logit），我们按照下式给出输出：
$$q_i = \frac{\exp(z_i/T)}{\sum_j \exp(z_j/T)}$$

其中，当$T = 1$时，就是普通的softmax变换。这里令$T > 1$，就得到了软化的softmax。（这个很好理解，除以一个比$1$大的数，相当于被squash了，线性的sqush被指数放大，差距就不会这么大了）。OK，有了这个东西，我们将Teacher网络和Student的最后充当分类器的那个全连接层的输出都做这个处理。

对Teacher网络的logit如此处理，得到的就是soft target。相比于one-hot的ground truth或softmax的prob输出，这个软化之后的target能够提供更多的类别间和类内信息。
可以对待训练的Student网络也如此处理，这样就得到了另外一个“交叉熵”损失：

$$\mathcal{L}_{soft}=-\sum_{i=1}^{K}p_i\log q_i$$

其中，$p_i$为Teacher模型给出的soft target，$q_i$为Student模型给出的soft output。作者发现，最好的方式是做一个multi task learning，将上面这个损失函数和真正的交叉熵损失加权相加。相应地，我们将其称为hard target。

$$\mathcal{L} = \mathcal{L}_{hard} + \lambda \mathcal{L}_{soft}$$

其中，$\mathcal{L}_{hard}$是分类问题中经典的交叉熵损失。由于做softened softmax计算时，需要除以$T$，导致soft target关联的梯度幅值被缩小了$T^2$倍，所以有必要在$\lambda$中预先考虑到$T^2$这个因子。

PS:这里有一篇地平线烫叔关于多任务中loss函数设计的回答：[神经网络中，设计loss function有哪些技巧? - Alan Huang的回答 - 知乎](https://www.zhihu.com/question/268105631/answer/335246543)。

## 实现
这里给出一个开源的MXNet的实现:[kd loss by mxnet](https://github.com/TuSimple/neuron-selectivity-transfer/blob/master/symbol/transfer.py#L4)。MXNet中的`SoftmaxOutput`不仅能直接支持one-hot编码类型的array作为label输入，甚至label的`dtype`也可以不是整型！

``` py
def kd(student_hard_logits, teacher_hard_logits, temperature, weight_lambda, prefix):
    student_soft_logits = student_hard_logits / temperature
    teacher_soft_logits = teacher_hard_logits / temperature
    teacher_soft_labels = mx.symbol.SoftmaxActivation(teacher_soft_logits,
        name="teacher%s_soft_labels" % prefix)
    kd_loss = mx.symbol.SoftmaxOutput(data=student_soft_logits, label=teacher_soft_labels,
                                      grad_scale=weight_lambda, name="%skd_loss" % prefix)
    return kd_loss

```

## matching logit是特例
（这部分没什么用，练习推导了一下交叉熵损失的梯度计算）

在Hinton之前，有学者提出可以匹配Teacher和Student输出的logit，Hinton指出这是本文方法在一定假设下的近似。为了和论文中的符号相同，下面我们使用$C$表示soft target带来的loss，Teacher和Student第$i$个神经元输出的logit分别为$v_i$和$z_i$，输出的softened softmax分别为$p_i$和$q_i$。那么我们有：
$$C = -\sum_{j=1}^{C}p_j \log q_j$$

而且，
$$p_i = \frac{\exp(v_i/T)}{\sum_j \exp(v_j/T)}$$
$$q_i = \frac{\exp(z_i/T)}{\sum_j \exp(z_j/T)}$$

让我们暂时忽略$T$（最后我们乘上$\frac{1}{T}$即可），我们有：
$$\frac{\partial C}{\partial z_i} = -\sum_{j=1}^{K}p_j\frac{1}{q_j}\frac{\partial q_j}{\partial z_i}$$

分情况讨论，当$i = j$时，有：
$$\frac{\partial q_j}{\partial z_i} = q_i (1-q_i)$$

当$i \neq j$时，有：
$$\begin{aligned}
\frac{\partial q_j}{\partial z_i} &= \frac{-e^{z_i}e^{z_j}}{(\sum_k e^{z_k})^2}  \\
&=-q_iq_j
\end{aligned}$$

这样，我们有：
$$
\begin{aligned} 
\frac{\partial C}{\partial z_i} &= - p_i\frac{1}{q_i}q_i(1-q_i) + \sum_{j=1, j\neq i}^{K}p_j\frac{1}{q_j}q_iq_j  \\
&= -p_i + p_iq_i + \sum_{j=1, j\neq i}^K p_jq_i \\
&= q_i -p_i
\end{aligned}$$
当然，其实上面的推导过程只不过是重复了一遍one-hot编码的交叉熵损失的计算。

这样，如果我们假设logit是零均值的，也就是说$\sum_j z_j = \sum_j v_j = 0$，那么有：
$$\frac{\partial C}{\partial z_i} \sim \frac{1}{NT^2}(z_i - v_i)$$

所以说，MSE下进行logit的匹配，是本文方法的一个特例。

## 实验
作者使用了MNIST进行图片分类的实验，一个有趣的地方在于（和论文前半部分举的$2$和$3$识别的例子呼应），作者在数据集中有意地去除了标签为$3$的样本。没有KD的student网络不能识别测试时候提供的$3$，有KD的student网络能够识别一些$3$（虽然它从来没有在训练样本中出现过！）。后面，作者在语音识别和一个Google内部的很大的图像分类数据集（JFT dataset）上做了实验，

## 附
- 知乎上关于soft target的讨论，有Wang Naiyan和Zhou Bolei的分析：[如何理解soft target这一做法？
](https://www.zhihu.com/question/50519680)

