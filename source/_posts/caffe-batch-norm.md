---
title: Caffe中的BatchNorm实现
date: 2018-01-08 20:12:44
tags:
     - cafe
---
这篇博客总结了Caffe中BN的实现。
<!-- more -->

## BN简介

由于BN技术已经有很广泛的应用，所以这里只对BN做一个简单的介绍。

BN是Batch Normalization的简称，来源于Google研究人员的论文：[Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/pdf/1502.03167.pdf)。对于网络的输入层，我们可以采用减去均值除以方差的方法进行归一化，对于网络中间层，BN可以实现类似的功能。

在BN层中，训练时，会对输入blob各个channel的均值和方差做一统计。在做inference的时候，我们就可以利用均值和方法，对输入$x$做如下的归一化操作。其中，$\epsilon$是为了防止除数是$0$，$i$是channel的index。
$$\hat{x_i} = \frac{x_i-\mu_i}{\sqrt{Var(x_i)+\epsilon}}$$

不过如果只是做如上的操作，会影响模型的表达能力。例如，Identity Map($y = x$)就不能表示了。所以，作者提出还需要在后面添加一个线性变换，如下所示。其中，$\gamma$和$\beta$都是待学习的参数，使用梯度下降进行更新。BN的最终输出就是$y$。
$$y_i = \gamma \hat{x_i} + \beta$$

如下图所示，展示了BN变换的过程。
![BN变换](/img/caffe_bn_what_is_bn.jpg)

上面，我们讲的还是inference时候BN变换是什么样子的。那么，训练时候，BN是如何估计样本均值和方差的呢？下面，结合Caffe的代码进行梳理。

## BN in Caffe
在BVLC的Caffe实现中，BN层需要和Scale层配合使用。在这里，BN层专门用来做“Normalization”操作（确实是人如其名了），而后续的线性变换层，交给Scale层去做。

下面的这段代码取自He Kaiming的Residual Net50的[模型定义文件](https://github.com/KaimingHe/deep-residual-networks/blob/master/prototxt/ResNet-50-deploy.prototxt#L21)。在这里，设置`batch_norm_param`中`use_global_stats`为`true`，是指在inference阶段，我们只使用已经得到的均值和方差统计量，进行归一化处理，而不再更新这两个统计量。后面Scale层设置的`bias_term: true`是不可省略的。这个选项将其配置为线性变换层。

```
layer {
	bottom: "conv1"
	top: "conv1"
	name: "bn_conv1"
	type: "BatchNorm"
	batch_norm_param {
		use_global_stats: true
	}
}

layer {
	bottom: "conv1"
	top: "conv1"
	name: "scale_conv1"
	type: "Scale"
	scale_param {
		bias_term: true
	}
}
```
这就是Caffe中BN层的固定搭配方法。这里只是简单提到，具体参数的意义待我们深入代码可以分析。

## BatchNorm 层的实现
上面说过，Caffe中的BN层与原始论文稍有不同，只是做了输入的归一化，而后续的线性变换是交由后续的Scale层实现的。我们首先看一下`caffe.proto`中关于BN层参数的描述。保留了原始的英文注释，并添加了中文解释。
```
message BatchNormParameter {
  // If false, normalization is performed over the current mini-batch
  // and global statistics are accumulated (but not yet used) by a moving
  // average.
  // If true, those accumulated mean and variance values are used for the
  // normalization.
  // By default, it is set to false when the network is in the training
  // phase and true when the network is in the testing phase.
  // 设置为False的话，更新全局统计量，对当前的mini-batch进行规范化时，不使用全局统计量，而是
  // 当前batch的均值和方差。
  // 设置为True，使用全局统计量做规范化。
  // 后面在BN的实现代码我们会看到，这个变量默认随着当前网络在train或test phase而变化。
  // 当train时为false，当test时为true。
  optional bool use_global_stats = 1;
  
  // What fraction of the moving average remains each iteration?
  // Smaller values make the moving average decay faster, giving more
  // weight to the recent values.
  // Each iteration updates the moving average @f$S_{t-1}@f$ with the
  // current mean @f$ Y_t @f$ by
  // @f$ S_t = (1-\beta)Y_t + \beta \cdot S_{t-1} @f$, where @f$ \beta @f$
  // is the moving_average_fraction parameter.
  // BN在统计全局均值和方差信息时，使用的是滑动平均法，也就是
  // St = (1-beta)*Yt + beta*S_{t-1}
  // 其中St为当前估计出来的全局统计量（均值或方差），Yt为当前batch的均值或方差
  // beta是滑动因子。其实这是一种很常见的平滑滤波的方法。
  optional float moving_average_fraction = 2 [default = .999];
  
  // Small value to add to the variance estimate so that we don't divide by
  // zero.
  // 防止除数为0加上去的eps
  optional float eps = 3 [default = 1e-5];
}
```

OK。现在可以进入BN的代码实现了。阅读大部分代码都没有什么难度，下面主要结合代码讲解`use_global_stats`变量的作用和均值（方差同理）的计算。由于均值和方差的计算原理相近，所以下面只会详细介绍均值的计算。


