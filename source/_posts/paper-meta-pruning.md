---
title: 论文 - MetaPruning：Meta Learning for Automatic Neural Network Channel Pruning
date: 2019-10-26 00:33:37
tags:
     - paper
     - model compression
---

这篇文章来自于旷视。旷视内部有一个基础模型组，孙剑老师也是很看好NAS相关的技术，相信这篇文章无论从学术上还是工程落地上都有可以让人借鉴的地方。回到文章本身，模型剪枝算法能够减少模型计算量，实现模型压缩和加速的目的，但是模型剪枝过程中确定剪枝比例等参数的过程实在让人头痛。这篇文章提出了PruningNet的概念，自动为剪枝后的模型生成权重，从而绕过了费时的retrain步骤。并且能够和进化算法等搜索方法结合，通过搜索编码network的coding vector，自动地根据所给约束搜索剪枝后的网络结构。和AutoML技术相比，这种方法并不是从头搜索，而是从已有的大模型出发，从而缩小了搜索空间，节省了搜索算力和时间。个人觉得这种剪枝和NAS结合的方法，应该会在以后吸引越来越多人的注意。这篇文章的代码已经开源在了Github：[MetaPruning](https://github.com/liuzechun/MetaPruning)。

这篇文章首发于[Paper Weekly公众号](https://wemp.app/accounts/fd027dce-bcd1-4eaf-9e64-88bffd7ca8a2)，欢迎关注。

<!-- more -->

## Motivation

模型剪枝是一种能够减少模型大小和计算量的方法。模型剪枝一般可以分为三个步骤：

- 训练一个参数量较多的大网络
- 将不重要的权重参数剪掉
- 剪枝后的小网络做fine tune

其中第二步是模型剪枝中的关键。有很多paper围绕“怎么判断权重是否重要”以及“如何剪枝”等问题进行讨论。困扰模型剪枝落地的一个问题就是剪枝比例的确定。传统的剪枝方法常常需要人工layer by layer地去确定每层的剪枝比例，然后进行fine tune，用起来很耗时，而且很不方便。不过最近的[Rethinking the Value of Network Pruning](https://arxiv.org/abs/1810.05270)指出，剪枝后的权重并不重要，对于channel pruning来说，更重要的是找到剪枝后的网络结构，具体来说就是每层留下的channel数量。受这个发现启发，文章提出可以用一个PruningNet，对于给定的剪枝网络，自动生成weight，无需进行retrain，然后评测剪枝网络在验证集上的性能，从而选出最优的网络结构。

具体来说，PruningNet的输入是剪枝后的网络结构，必须首先对网络结构进行编码，转换为coding vector。这里可以直接用剪枝后网络每层的channel数来编码。在搜索剪枝网络的时候，我们可以尝试各种coding vector，用PruningNet生成剪枝后的网络权重。网络结构和权重都有了，就可以去评测网络的性能。进而用进化算法搜索最优的coding vector，也就是最优的剪枝结构。在用进化算法搜索的时候，可以使用自定义的目标函数，包括将网络的accuracy，latency，FLOPS等考虑进来。

![PruningNet的训练和使用](/img/paper_metapruning_pruningnet.jpg)

## PruningNet

从上一小节已经可以知道，PruningNet是整个算法的关键。那么怎么才能找到这样一个“神奇网络”呢？

先做一下符号约定，使用$c_i$表示剪枝之后第$i$层的channel数量，$l$为网络的层数，$W$表示剪枝后网络的权重。那么PruningNet的输入输出如下所示：

$W = \text{PruningNet}(c_1, c_2, \dots, c_l)$

### 训练

先结合下图看一下forward部分。PruningNet是由$l$个PruningBlock组成的，每个PruningBlock是一个两层的MLP。首先看图b，编码着网络结构信息的coding vector输入到当前block后，输出经过Reshape，成了一个Weight Matrix。注意哦，这里的WeightMatrix是固定大小的（也就是未剪枝的原始Weight shape大小），和剪枝网络结构无关。再看图a，因为要对网络进行剪枝，所以WeightMatrix要进行Crop。对应到图b，可以看到，Crop是在两个维度上进行的。首先，由于上一层也进行了剪枝，所以input channel数变少了；其次，由于当前层进行了剪枝，所以output channel数变少了。这样经过Crop，就生成了剪枝后的网络weight。我们再输入一个mini batch的训练图片，就可以得到剪枝后的网络的loss。

![PruningNet train forward](/img/paper_metapruning_pruningnet_forward.jpg)

在backward部分，我们不更新剪枝后网络的权重，而是更新PruningNet的权重。由于上面的操作都是可微分的，所以直接用链式法则传过去就行。如果你使用PyTorch等支持自动微分的框架，这是很容易的。

下图所示是训练过程的整个PruningNet（左侧）和剪枝后网络（右侧，即PrunedNet）。训练过程中的coding vector在状态空间里随机采样，随机选取每层的channel数量。

PS：和原始论文相比，下图和上图顺序是颠倒的。这里从底向上介绍了PruningNet的训练，而论文则是自顶向下。

![整个PruningNet](/img/paper_metapruning_whole_meta_learning.jpg)

### 搜索

训练好PruningNet后，就可以用它来进行搜索了！我们只需要输入某个coding vector，PruningNet就会为我们生成对应每层的WeightMatrix。别忘了coding vector是编码的网络结构，现在又有了weight，我们就可以在验证集上测试网络的性能了。进而，可以使用进化算法等优化方法去搜索最优的coding vector。当我们得到了最优结构的剪枝网络后，再from scratch地训练它。

进化算法这里不再赘述，很多优化的书中包括网上都有资料。这里把整个算法流程贴出来：

![进化算法流程](/img/paper_metapruning_evaluation_algorithm.jpg)

## 实验

作者在ImageNet上用MobileNet和ResNet进行了实验。训练PruningNet用了$\frac{1}{4}$的原模型的epochs。数据增强使用常见的标准流程，输入image大小为$224\times 224$。

将原始ImageNet的训练集做分割，每个类别选50张组成sub-validation（共计50000），其余作为sub-training。在训练时，我们使用sub-training训练PruningNet。在搜索时，使用sub-validation评估剪枝网络的性能。不过，还要注意，在搜索时，使用20000张sub-training中的图片重新计算BatchNorm layer中的running mean和running variance。

### shortcut剪枝

在进行模型剪枝时，一个比较难处理的问题是ResNet中的shortcut结构。因为最后有一个element-wise的相加操作，必须保证两路feature map是严格shape相同的，所以不能随意剪枝，否则会造成channel不匹配。下面对几种论文中用到的网络结构分别讨论。

#### MobileNet-v1

MobileNet-v1是没有shortcut结构的。我们为每个conv layer都配上相应的PruningBlock——一个两层的MLP。PruningNet的输入coding vector中的元素是剪枝后每层的channel数量。而输入第$i$个PruningBlock的是一个2D vector，由归一化的第$i-1$层和第$i$层的剪枝比例构成。这部分可以结合代码[MetaPruning](https://github.com/liuzechun/MetaPruning/blob/master/mobilenetv1/training/mobilenet_v1.py#L15)来看。注意第$1$个conv layer的输入是1D vector，因为它是第一个被剪枝的layer。在训练时，coding vector的搜索空间被以一定步长划分为grid，采样就是在这些格点上进行的。

#### MobileNet-v2

MobileNet-v2引入了类似ResNet的shortcut结构，这种resnet block必须统一看待。具体来说，对于没有在resnet block中的conv，处理方法如MobileNet-v1。对每个resnet block，配上一个相应的PruningBlock。由于每个resnet block中只有一个中间层（$3\times 3$的conv），所以输出第$i$个PruningBlock的是一个3D vector，由归一化的第$i-1$个resnet block，第$i$个resnet block和中间conv层的剪枝比例构成。其他设置和MobileNet-v1相同。这里可以结合代码[MetaPruning](https://github.com/liuzechun/MetaPruning/blob/master/mobilenetv2/training/mobilenet_v2.py#L109)来看。

#### ResNet

处理方法如MobileNet-v2所示。可以结合代码[MetaPruning](https://github.com/liuzechun/MetaPruning/blob/master/resnet/training/resnet.py#L75)来看。

### 实验结果

在相近FLOPS情况下，和MobileNet论文中改变ratio参数得到的模型比较，MetaPruning得到的模型accuracy更高。尤其是压缩比例更大时，该方法更有优势。

![MobileNet baseline比较](/img/paper_metapruning_compare_with_mobilenet_baseline.jpg)

和其他剪枝方法（如[AMC](https://arxiv.org/abs/1802.03494)）等方法比较，该方法也得到了SOTA的结果。MetaPruning方法能够以一种统一的方法处理ResNet中的shortcut结构，并且不需要人工调整太多的参数。

![和其他剪枝方法比较](/img/paper_metapruning_compare_with_other_pruning_automl.jpg)

上面的比较都是基于理论FLOPS，现在更多人在关注网络在实际硬件上的latency怎么样。文章对此也进行了讨论。如何测试网络的latency？当然可以每个网络都实际跑一下，不过有些麻烦。基于每个layer的inference时间是互相独立的这个假设，作者首先构造了各个layer inference latency的查找表（参见论文[Fbnet: Hardware-aware efficient convnet design via differentiable neural architecture search](https://arxiv.org/abs/1812.03443)），以此来估计实际网络的latency。作者这里和MobileNet baseline做了比较，结果也证明了该方法更优。

![latency比较](/img/paper_metapruning_latency_compare_with_mobilenet_baseline.jpg)

### PruningNet结果分析

此外，作者还对PruningNet的预测结果进行可视化，试图找出一些可解释性，并找出剪枝参数的一些规律。

- down-sampling的部分PruningNet倾向于保留更多的channel，如MobileNet-v2 block中间的那个conv
- 优先剪浅层layer的channel，FLOPS约束太强剪深层的channel，但可能会造成网络accuracy下降比较多

## 结论

这篇文章从“剪枝后的weight作用不大”的现象出发，将剪枝和NAS结合，提出了PruningNet为剪枝后的网络预测weight，避免了网络的retrain，从而可以快速衡量剪枝网络的性能。并在编码网络信息的coding vector状态空间进行搜索，找到给定约束条件下的最优网络结构，在ImageNet数据集和ResNet/MobileNet-v1/v2上取得了比之前剪枝算法更好的效果。
