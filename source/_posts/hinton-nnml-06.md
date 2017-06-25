---
title: Neural Network for Machine Learning - Lecture 06
date: 2017-06-25 13:48:31
tags:
    - 公开课
    - deep learning
---
第六周的课程主要讲解了用于神经网络训练的梯度下降方法，首先对比了SGD，full batch GD和mini batch SGD方法，然后给出了几个用于神经网络训练的trick，主要包括输入数据预处理（零均值，单位方差以及PCA解耦），学习率的自适应调节以及网络权重的初始化方法（可以参考各大框架中实现的Xavier初始化方法等）。这篇文章主要记录了后续讲解的几种GD变种方法，如何合理利用梯度信息达到更好的训练效果。

<!-- more -->

## Momentum
我们可以把训练过程想象成在权重空间的一个质点（小球），移动到全局最优点的过程。不同于GD，使用梯度信息直接更新权重的位置，momentum方法是将梯度作为速度量。这样做的好处是，当梯度的方向一直不变时，速度可以加快；当梯度方向变化剧烈时，由于符号改变，所以速度减慢，起到了GD中自适应调节学习率的过程。

具体来说，我们利用新得到的梯度信息，采用滑动平均的方法更新速度。式子中的$\epsilon$为学习率，$\alpha$为momentum系数。
$$\Delta w_t = v_t = \alpha v_{t-1} - \epsilon g_t = \Delta w_t - \epsilon g_t$$

为了说明momentum确实对学习过程有加速作用，假设一个简单的情形，即运动轨迹是一个斜率固定的斜面。那么我们有梯度$g$固定。根据上面的递推公式可以得到通项公式（简单的待定系数法凑出等比数列）：
$$v_t = \alpha(v_{t-1} + \frac{\epsilon g}{1-\alpha}) - \frac{\epsilon g}{1-\alpha}$$

由于$\alpha < 0$，所以当$t = \infty$时，只剩下了后面的常数项，即：
$$v_\infty = -\frac{\epsilon}{1-\alpha}g$$

也就是说，权重更新的幅度变成了原来的$\frac{1}{1-\alpha}$倍。若取$\alpha=0.99$，则加速$100$倍。

Hinton给出的建议是由于训练开头梯度值比较大，所以momentum系数一开始不要过大，例如可以取$0.5$。当梯度值较小，训练过程被困在一个峡谷的时候，可以适当提升。

一种改进方法由Nesterov提出。在上面的方法中，我们首先更新了在该处的累积梯度信息，然后向前移动。而Nesterov方法中，我们首先沿着累计梯度信息向前走，然后根据梯度信息进行更正。

![Nesterov方法](/img/hinton_06_nesterov_momentum.png)

## Adaptive Learning Rate
这种方法起源于这样的观察：在网络中，不同layer之间的权重更新需要不同的学习率。因为浅层和深层的layer梯度幅值很可能不同。所以，对不同的权重乘上不同的因子是个更加合理的选择。

例如，我们可以根据梯度是否发生符号变化按照下面的方式调节某个权重$w_{ij}$的增益。注意$0.95$和$0.05$的和是$1$。这样可以使得平衡点在$1$附近。
![Different learning rate gain](/img/hinton_06_learningrate.png)

下面是使用这种方法的几个trick，包括限幅，较大的batch size以及和momentum的结合。

![Tricks for adaptive lr](/img/hinton_06_tricks_for_adaptive_lr.png)

## RMSProp
rprop利用梯度的符号，如果符号保持不变，则相应增大step size；否则减小。但是只能用于full batch GD。RMSProp就是一种可以结合mini batch SGD和rprop的一种方法。

我们使用滑动平均方法更新梯度的mean square（即RMS中的MS得来）。

$$\text{MeanSquare}(w, t) = 0.9 \text{MeanSquare}(w, t-1) + 0.1g_t^2$$

然后，将梯度除以上面的得到的Mean Square值。

RMSProp还有一些变种，列举如下：
![Otehr RMSProp](/img/hinton_06_rmsprop_improvement.png)

## 总结
- 对于小数据集，使用full batch GD（LBFGS或adaptive learning rate如rprop）。
- 对于较大数据集，使用mini batch SGD。并可以考虑加上momentmum和RMSProp。

如何选择学习率是一个较为依赖经验的任务（网络结构不同，任务不同）。
![总结](/img/hinton_06_summary.png)

此外，[这里](https://arxiv.org/abs/1609.04747)有一篇不错的各种学习方法的总结，可以一看。
