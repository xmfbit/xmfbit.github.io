---
title: 论文 - SqueezeNet, AlexNet-level accuracy with 50x fewer parameters and <0.5MB model size
date: 2018-03-24 14:02:53
tags:
---
[SqueezeNet](https://arxiv.org/abs/1602.07360)由HanSong等人提出，和AlexNet相比，用少于$50$倍的参数量，在ImageNet上实现了comparable的accuracy。比较本文和HanSoing其他的工作，可以看出，其他工作，如Deep Compression是对已有的网络进行压缩，减小模型size；而SqueezeNet是从网络设计入手，从设计之初就考虑如何使用较少的参数实现较好的性能。可以说是模型压缩的两个不同思路。

<!-- more -->
## 模型压缩相关工作
模型压缩的好处主要有以下几点：
- 更好的分布式训练。server之间的通信往往限制了分布式训练的提速比例，较少的网络参数能够降低对server间通信需求。
- 云端向终端的部署，需要更低的带宽，例如手机app更新或无人车的软件包更新。
- 更易于在FPGA等硬件上部署，因为它们往往都有着非常受限的片上RAM。

相关工作主要有两个方向，即模型压缩和模型结构自身探索。

模型压缩方面的工作主要有，使用SVD分解，Deep Compression等。模型结构方面比较有意义的工作是GoogLeNet的Inception module（可在博客内搜索*Xception*发现Xception的作者是如何受此启发发明Xception结构的）。


本文的作者从网络设计角度出发，提出了名为SqueezeNet的网络结构，使用比AlexNet少$50$倍的参数，在ImageNet上取得了comparable的结果。此外，还探究了CNN的arch是如何影响model size和最终的accuracy的。主要从两个方面进行了探索，分别是*CNN microarch*和*CNN macroarch*。前者意为在更小的粒度上，如每一层的layer怎么设计，来考察；后者是在更为宏观的角度，如一个CNN中的不同layer该如何组织来考察。

## SqueezeNet
为了简单，下文简称*SNet*。SNet的基本组成是叫做*Fire*的module。我们知道，对于一个CONV layer，它的参数数量计算应该是：$K \time K \times M \times N$。其中，$K$是filter的spatial size，$M$和$N$分别是输入feature map和输出activation的channel size。由此，设计SNet时，作者的依据主要是以下几点：
- 把$3\times 3$的卷积替换成$1\times 1$，相当于减小上式中的$K$。
- 减少$3\times 3$filter对应的输入feature map的channel，相当于减少上式的$M$。
- delayed downsample。使得activation的feature map能够足够大，这样对提高accuracy有益。CNN中的downsample主要是通过CONV layer或pooling layer中stride设置大于$1$得到的，作者指出，应将这种操作尽量后移。

> Our intuition is that large activation maps (due to delayed downsampling) can lead to higher classification accuracy, with all else held equal.

### Fire Module
Fire Module是SNet的基本组成单元，如下图所示。可以分为两个部分，一个是上面的*squeeze*部分，是一组$1\times 1$的卷积，用来将输入的channel squeeze到一个较小的值。后面是*expand*部分，由$1\times 1$和$3\times 3$卷积mix起来。使用$s_{1 x 1}$，$e_{1x1}$和$e_{3x3}$表示squeeze和expand中两种不同卷积的channel数量，令$s_{1x1} < e_{1x1} + e_{3x3}$，用来实现上述策略2.
![Fire Module示意](/img/paper-squeezenet-fire-module.png)

下面，对照PyTorch实现的SNet代码看下Fire的实现，注意上面说的CONV后面都接了ReLU。
``` py
class Fire(nn.Module):
    def __init__(self, inplanes, squeeze_planes,
                 expand1x1_planes, expand3x3_planes):
        super(Fire, self).__init__()
        self.inplanes = inplanes
        ## squeeze 部分
        self.squeeze = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1)
        self.squeeze_activation = nn.ReLU(inplace=True)
        ## expand 1x1 部分
        self.expand1x1 = nn.Conv2d(squeeze_planes, expand1x1_planes,
                                   kernel_size=1)
        self.expand1x1_activation = nn.ReLU(inplace=True)
        ## expand 3x3部分
        self.expand3x3 = nn.Conv2d(squeeze_planes, expand3x3_planes,
                                   kernel_size=3, padding=1)
        self.expand3x3_activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.squeeze_activation(self.squeeze(x))
        ## 将expand 部分1x1和3x3的cat到一起
        return torch.cat([
            self.expand1x1_activation(self.expand1x1(x)),
            self.expand3x3_activation(self.expand3x3(x))], 1)
```

