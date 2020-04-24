layout: post
title: 论文 - Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference
date: 2020-04-15 14:19:42
tags:
    - paper
    - model compression
    - model quantization
---

Google比较早的关于training-aware-quantization的模型量化的paper，不过提供了很多模型量化的基本知识。后面不管是TFLite还是TensorRT，都能在这篇文章中找到对应的基础知识。Arxiv: [Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference](https://arxiv.org/abs/1712.05877)

<!-- more -->

## Quantization Scheme - 如何量化

在这个小节，我们更考虑如何在数学上去设计量化的conv操作。在下一个小节，会从更实际的角度来考虑一个CNN的量化。

把量化值$q$映射到浮点数$r$的式子很简单，其中$S$和$Z$是量化参数，分别为零点（zero-point）和scaling factor：

$$r = S(q - Z)$$

$S$和$Z$作为量化参数，每个weight array或者activation array是共享的，不同的array之间不共享。

这里的$S$决定了分辨率，也就是最小量化误差。而$Z$可以用来平衡掉数据偏离$0$的bias。

有了上述变换规则， CNN中常见的矩阵相乘可以表示为下图。其中，$\alpha$的值为$1,2,3$，分别表示输入矩阵1，输入矩阵2，结果矩阵3。式4给出了在量化后的$Q$上计算原始浮点$R$的矩阵的方法。$\Sigma$内都是整数运算，只需要最后乘上一个scaling factor $M$即可，从而加速了计算。

![矩阵乘法](/img/paper_tf_taq_matmul.png)

## CNN的量化 - 以Conv为例

以卷积op为例，说明在实际的CNN模型中量化是如何做的。

### layer fusion

首先要做的是layer fusion，例如把常见的conv + bn + relu的3个op简化为一个op。relu这种op的fusion不多说，BN的fusion需要考虑权重，参见下个小节。

卷积实际上还是在进行矩阵乘法。在相乘的时候，用`uint8`来存储两个操作数，用`int32`存储结果，以防止相加的时候溢出。如：

```
int32 += uint8 * uint8
```

对于bias，也使用`int32`存储。作者在这里指出，使用较高精度存储bias，是很有必要的。因为bias的每个值都要加到对应channel的所有activation上去，所以它的比较小的量化误差也会对结果造成比较大的影响。综合起来，对于bias，我们使用$S = S\_1S\_2$（和conv的结果的scaling factor相同），且zero-point为$0$。

如下图所示，当conv / +bias / relu等操作做完之后，再进行量化。最终完成了fusion之后的conv op的计算。

![conv的计算](/img/paper_tf_taq_conv_quantization.png)

## Training-aware Quantization

在训练时，是使用float精度模拟量化模型，并使用float更新梯度。在inference的时候，直接在支持INT8的硬件上跑inference。这时候就是原生的量化模型在前向计算了。如下图所示：

![train && inference](/img/paper_tf_taq_train_inference.png)

这里作者首先分析了量化模型相对于原始精度模型可能的掉点原因：

- 同一组weight或activation的不同channel之间的差异较大。因为我们上面已经说过，它们会共享同一个scaling factor。所以如果某个channel的weight特别小，就会造成相对误差很大。
- 某些离群点(outlier)影响，把整个分布带偏。

要使用float模拟定点量化模型前向传播，要特别注意量化究竟是在哪里发生的。要注意的是下面两种情况：

- 对于weight来说，如果有BN的话，要先把BN和conv的weight做fusion，再量化，再前向计算，这里下面会详细说明一下
- 对于activation来说，一般是激活函数之后，还有bypass的相加或concat之后（对ResNet这种结构）

### BN的fusion

前面提到，BN要和conv fusion到一起。按channel做如下操作即可：

$$w_{\text{fold}} = \frac{\gamma w}{\sqrt{\sigma^2 + \epsilon}}$$
 
再加上量化，训练时候的整个计算图如下所示：

![conv/bn folding](/img/paper_tf_taq_conv_bn_folding_in_taining.png)


### 带饱和的量化

为了避免outlier的影响，在量化之前要先进行一波饱和操作，然后将值域均匀地映射到定点数表示的范围，例如8bit量化为256个stage。

![quantization_with_clamp](/img/tf_paper_taq_quantize_with_clamp.png)

不过新的问题出现了。对于weight，可以很容易地找到这样的$a$和$b$，但是对于activation，只有模型跑起来才能知道其范围。文章指出可以使用指数滑动平均来做（就是BN在训练时更新moving_mean和moving_variance的方式），要注意的点在于：

- 滑动平均的系数应该定的很接近$1$，有利于变化比较平滑
- 开始的若干步训练，可以暂时去掉activation的quantization，有利于网络稳定