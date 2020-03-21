---
title: 论文 - Bag of Tricks for Image Classification with Convolutional Neural Networks
date: 2019-07-06 13:52:45
tags:
---
这是[Bag of Tricks for Image Classification with Convolutional Neural Networks](https://arxiv.org/abs/1812.01187)的笔记。这篇文章躺在阅读列表里面很久了，里面的技术之前也用了一些。最近趁着做SOTA模型的训练，把论文整体读了一下，记录在这里。这篇文章总结的仍然是在通用学术数据集上的tricks。对于实际工作中遇到的训练任务，仍然是要结合问题本身来改进模型和训练算法。毕竟，没有银弹。

![软工里面没有银弹，数据科学同样这样](/img/bag_of_tricks_no_silver_bullet.jpeg)

<!-- more -->

## 简介

这篇文章主要讨论了训练图片分类模型的tricks，包括data augmentation，（lr，batch size等）超参设置，模型架构微调和模型蒸馏等技术。可以在增加少许计算量的情况下，把ResNet-50的top 1 acc提升4个点，从而打败许多后起之秀。Talk is cheap, show me the code. 论文讨论的方法对应代码，都已经在GluonCV中开源，所以建议在阅读论文的时候，对照[代码](https://github.com/dmlc/gluon-cv/blob/master/scripts/classification/imagenet/train_imagenet.py)进行学习。

![ResNet-50的效果提升](/img/bag_of_tricks_resnet50_overperform_others.jpg)

## Baseline Training

这里介绍了一些（已经不算trick的）训练ResNet-50可以注意的地方。使用这些方法，应该可以复现论文中给出的结果。

### Data Argumentation

这里都是老生常谈了，可以直接参看代码[gluon cv/image classification](https://github.com/dmlc/gluon-cv/blob/master/scripts/classification/imagenet/train_imagenet.py#L203)。

``` py
jitter_param = 0.4
lighting_param = 0.1
mean_rgb = [123.68, 116.779, 103.939]
std_rgb = [58.393, 57.12, 57.375]
train_data = mx.io.ImageRecordIter(
    path_imgrec         = rec_train,
    path_imgidx         = rec_train_idx,
    preprocess_threads  = num_workers,
    shuffle             = True,
    batch_size          = batch_size,
    data_shape          = (3, input_size, input_size),
    mean_r              = mean_rgb[0],
    mean_g              = mean_rgb[1],
    mean_b              = mean_rgb[2],
    std_r               = std_rgb[0],
    std_g               = std_rgb[1],
    std_b               = std_rgb[2],
    rand_mirror         = True,
    random_resized_crop = True,
    max_aspect_ratio    = 4. / 3.,
    min_aspect_ratio    = 3. / 4.,
    max_random_area     = 1,
    min_random_area     = 0.08,
    brightness          = jitter_param,
    saturation          = jitter_param,
    contrast            = jitter_param,
    pca_noise           = lighting_param,
)
```

### 参数初始化

- 使用Xavier初始化卷积层和全连接层的权重，也就是$w\sim \mathcal{U}(-a, a)$，其中$a = \sqrt{6/(d\_{in} + d\_{out})}$，$d$是输入和输出的channel size。偏置项初始化为$0$。
- BatchNorm的$\gamma$初始化为$1$，偏置项为$0$。

### 训练参数

8卡V100，batch size = 256，使用NAG梯度下降，lr从0.1，在30，60，90epoch处除以10。

使用上述设置，得到的ResNet-50模型比原始论文更好，不过Inception-V3（输入为$229\times 229$大小）和MobileNet稍差于原始论文。

## 更快地训练

主要讨论使用低精度（FP16）和大batch size对训练的影响。

### 大batch size

大的batch size经常会导致模型的val acc降低（一个简单的解释是，大batch size造成iteration次数减少，导致模型效果变差。当然，实际训练中，大batch size常常搭配较大的lr，所以问题并不是这么简单），可以考虑使用下面的方法解决这个问题。

#### （成比例）提高lr

上面说的iteration次数减少是一个方面。另一个考虑是大的batch size会造成对梯度的估计方差变小，我们可以乘上一个较大的lr，让方差的不确定性增大一些。一个经验之谈是，lr随着batch size成比例扩大。比如在训练ResNet-50的时候，He给出的在$B = 256$时，lr取为$0.1$。那么如果$B = 512$，那么lr也相应扩大为$0.2$。

#### lr warm up

如果lr初始设置的很大，可能会带来数值不稳定。因为刚开始的时候权重是随机初始化的，gradient也比较大。可以给lr做warm up，也就是开始若干个迭代用较小的lr，等训练稳定了再用回那个大的lr。一种方法是线性warm up，也就是在warm up阶段，lr是线性地从0涨到给定的那个大lr。

#### 设置$\gamma = 0$

这个操作比较新奇，在初始阶段，BN的$\beta$参数是设置为$0$的。如果我们再设置$\gamma = 0$，说明BN的输出就是$0$了。这是什么操作？！

作者指出，可以在ResNet这种有by-pass的结构中使用这个trick。在ResNet block的最后一层，我们经常做$y = x + res(x)$，可以考虑将res这一路的最后一个BN层的$\gamma$参数设置为0。这时候，相当于只有输入$x$传到后面，相当于减少了网络的层深。之后的训练中，$\gamma$会逐渐变大，也就逐渐恢复了res通路。

这种方法也是试图解决网络训练初始阶段不稳定的问题。不过这个操作还是挺骚的。。。类似的方法（利用BN层的$\gamma$参数）也见到过被用在模型剪枝上，如Net Sliming等方法。可以参见博客中的相关文章讨论。

#### weight decay

给weight加上L2 norm来做weight decay，是缓解网络过拟合的标准解决办法之一。不过，最好只对conv和fc的kernel做，而不要对它们的bias，BN的$\gamma$和$\beta$做。

上面的方法，在batch size不大于2K的时候，应该是够用了的。

### 低精度

很多新GPU都加入了FP16的硬件支持，例如V100上使用FP16比FP32，训练能够加速$2$到$3$倍。FP16的问题是表示范围变小了，同时分辨率变小。对应地会造成两个问题，溢出和无法更新（梯度过小，不到FP16的最小表示）。一种解决办法是使用FP16来做forward和backward，但是在FP32上更新梯度（防止梯度过小）。同时给loss乘上一个系数，让它更好地契合FP16能表示的数据范围。

这里简要介绍下FP16精度的相关内容。关于Nvidia GPU FP16的更多信息，可以参考[Nvidia文档混合精度训练](https://docs.nvidia.com/deeplearning/sdk/mixed-precision-training/index.html)。

#### FP16数据表示
FP16，顾名思义，就是使用16个bit表示浮点数。具体编码方式上，和FP32基本一致，只不过位数有了缩水。

> IEEE 754 standard defines the following 16-bit half-precision floating point format: 1 sign bit, 5 exponent bits, and 10 fractional bits.

TODO: FP32和FP16的比较

#### FP16 in MXNet

在MXNet中，使用混合精度训练还是挺简单的。具体可以参考[Mixed precision training using float16](https://mxnet.incubator.apache.org/versions/master/faq/float16.html)

下面是使用gluon训练时候要注意的几个地方：

``` py
## optimizer 开启混合精度选项
## 这会使optimizer为参数保存一份FP32拷贝，在上面进行梯度的更新，
## 防止梯度过小无法更新FP16
if opt.dtype != 'float32':
    optimizer_params['multi_precision'] = True
## net cast到给定的数值精度
net = get_model(model_name, **kwargs)
net.cast(opt.dtype)
## 训练过程中，将输入也cast到指定精度
while in_training:
    ## blablabla
    outputs = [net(X.astype(opt.dtype, copy=False)) for X in data]
    ## 计算loss也把label cast到指定精度
```

使用MXNet老的symbolic接口时候，因为静态图一旦写好就固定了，所以我们需要在建图的时候，考虑FP16精度。

- 在原始输入node后面接一个`cast` op，将FP32转成FP16。
- 最好在`SoftmaxOutput`之前，插入一个`cast` op，将FP16转回FP32，以便有更高的精度。
- `optimizer`打开`multi_precision`开关，这里和上面gluon是一致的。

``` py
## 建图
data = mx.sym.Variable(name="data")
if dtype == 'float16':
    data = mx.sym.Cast(data=data, dtype=np.float16)
# ... the rest of the network
net_out = net(data)
if dtype == 'float16':
    net_out = mx.sym.Cast(data=net_out, dtype=np.float32)
output = mx.sym.SoftmaxOutput(data=net_out, name='softmax')
## 优化器设置
optimizer = mx.optimizer.create('sgd', multi_precision=True, lr=0.01)
```

下面有几条额外的建议：
- FP16加速主要来源于新GPU上的Tensor Core计算$D = A * B + C$这种运算，且它们的维度是$8$的倍数。所以如果不满足$8$倍数这个条件，FP16的计算速度可能不会很快，或者说和FP32相比没多少优势。尤其是当你在CIFAR10这种输入图片size比较小的数据集上训练的时候。
- 针对上面这种情况，你可以使用`nvprof`工具来check是否Tensor Core被使用了，那些名字里面带有`s884cudnn`的操作就是了。
- 确保data io和preprocessing不要成为瓶颈，不然面对这些扯后腿的地方，FP16男默女泪。
- batch size最好设置为8的倍数，2的幂次是坠吼的。
- 如果GPU memory还算充足，可以设置`MXNET_CUDNN_AUTOTUNE_DEFAULT = 2`，来让MXNet有更多的测试来选用最快的卷积算法，代价就是更多的显存占用。
- 最好为BatchNorm和SoftmaxOutput使用FP32精度。Gluon里面这些都是自动的，MXNet中BN层是自动的，但是SoftmaxOutput需要自己设置一下，见上。

#### loss scaling

再说一下上面提到的loss scaling。

为啥要做loss scaling呢？主要是由于FP16的精度比较差，而能够表示的较大的数对于CNN网络来说又基本用不到（虽然说FP16的表示范围相比FP32已经缩水不少了），所以可能出现这样一种情形，loss对FP16 weight或activation求梯度，梯度太小，以至于FP16无法表示。那其实我们可以给loss乘上一个系数，放大gradient，以便FP16能够表示。在梯度更新之前，再把这个梯度scale回去，就可以了。如下图所示。

![FP16和FP32的range不匹配](/img/bag_of_tricks_fp16_range_dismatch.jpg)

使用gluon或MXNet设置loss scaling的方法如下：

``` py
## gluon
loss = gluon.loss.SoftmaxCrossEntropyLoss(weight=128)
optimizer = mx.optimizer.create('sgd',
                                multi_precision=True,
                                rescale_grad=1.0/128)
## mxnet
mxnet.sym.SoftmaxOutput(other_args, grad_scale=128.0)
optimizer = mx.optimizer.create('sgd',
                                multi_precision=True,
                                rescale_grad=1.0/128)
```

经验来看，对于Multibox SSD, R-CNN, bigLSTM and Seq2seq这些任务，loss scaling是比较有必要的。这里有个疑问，loss scaling应该是在训练过程中不断变化的，但上面的使用都是直接把loss scaling写死了（gluon还好，再手动给loss乘上一个因子），那如何修改loss scaling呢？后面指出可以使用constant的loss scaling（一般取2的幂次64，128等），但是不知道实际训练会不会有问题。[Nvidia guide](https://docs.nvidia.com/deeplearning/sdk/mixed-precision-training/index.html)中给出的建议是：

> If you encounter precision problems, it is beneficial to scale the loss up by 128, and scale the application of the gradients down by 128.

当然，最好的办法是自己看一下FP32 gradient的分布。

当当当。。。说了这么多，那么具体加速效果如何呢？使用batch size = $1024$，和batch size = $256$的baseline相比，从下表可知，三种不同的网络结构，分别加速了$1.6X$到$3X$，而且acc还涨了一些。

![加速效果](/img/bag_of_tricks_accelarate_training.jpg)

具体的acc影响的ablation实验如下。可以看到，只是使用lr线性增大的情况下，大（batch size的）网络稍逊于小（batch size的）网络。不过当使用上面几个技术综合来看的时候，大小网络的性能差异已经抹去了，而且大网络的训练速度更快。

![ablation实验结果](/img/bag_of_tricks_ablation_of_accelarate_train.jpg)

## 更好的网络

TODO: 未完待续