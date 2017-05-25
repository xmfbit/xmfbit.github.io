---
title: Neural Network for Machine Learning - Lecture 02
date: 2017-05-25 13:26:33
tags:
    - 公开课
    - deep learning
---

## Different neural network archs
### Feed-forward neural network 前馈神经网络
前馈神经网络可能是最常见的网络，主要由输入层，若干隐含层和输出层组成。一般，当隐含层数目超过$1$时，我们可以说网络是deep的。
![前馈网络](/img/hinton_02_feed_forward_nn.png)

### Recurrent network
RNN内部的节点之间存在有向的环，这使得它能够使用内部状态来对动态过程建模。RNN能力强大，但是不易训练。
![RNN](/img/hinton_02_recurrent_nn.png)

RNN常用来对序列进行建模（modeling squence）。这里有一篇[不错的介绍](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)。
![RNN modeling squences](/img/hinton_02_rnn_app.png)

### Symmetrically connected network
这种网络结构上很像RNN，但是它的节点之间的连接是对称的，意思是说由此到彼和由彼到此的权重相同。
