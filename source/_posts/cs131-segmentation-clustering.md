---
title: CS131-KMeans聚类
date: 2017-02-05 23:07:00
tags:
    - cs131
    - 公开课
---
[K-Means聚类](https://zh.wikipedia.org/wiki/K-平均算法)是把n个点（可以是样本的一次观察或一个实例）划分到$k$个聚类中，使得每个点都属于离他最近的均值（此即聚类中心）对应的聚类，以之作为聚类的标准，满足最小化聚类中心和属于该中心的数据点之间的平方距离和（Sum of Square Distance，SSD）。
$$\text{SSD} = \sum_{i=1}^{k}\sum_{x\in c_i}(x-c_i)^2$$
![K-Means Demo](/img/kmeans_demo.png)

<!-- more -->
## 目标函数
K-Means方法实际上需要确定两个参数，$c^\ast$和$\delta^\ast$。其中$c\_{i}^\ast$代表各个聚类中心的位置，$\delta\_{ij}^\ast$的取值为$\lbrace 0,1\rbrace$，代表点$x_j$是否被分到了第$i$个聚类中心。

那么，目标函数可以写成如下的形式。
$$c^\ast, \delta^\ast = \arg\min_{c,\delta} \frac{1}{N}\sum_{j=1}^{N}\sum_{i=1}^{k}\delta_{i,j}(c_i-x_j)^2$$

然而这样一来，造成了一种很尴尬的局面。一方面，要想优化$c\_i$，需要我们给定每个点所属的类；另一方面，优化$\delta\_{i,j}$，需要我们给定聚类中心。由于这两个优化变量之间是纠缠在一起的，K-Means算法如何能够达到收敛就变成了一个讨论“先有鸡还是先有蛋”的问题了。

实际使用中，K-Means的迭代过程，实际是EM算法的一个特例。

## 算法流程
K-Means算法的流程如下所示。
![K-Means算法流程](/img/kmeans_algorithm.png)

假设我们有$N$个样本点，$\lbrace x_1, \dots, x_N\rbrace, x_i\in\mathbb{R}^D$，并给出聚类数目$k$。

首先，随机选取一系列的聚类中心点$\mu_i, i = 1,\dots, k$。接着，我们按照距离最近的原则为每个数据点指定相应的聚类中心并计算新的数据点均值更新聚类中心。如此这般，直到收敛。

## 算法细节
### 初始化
上面的K-Means方法对初始值很敏感。朴素的初始化方法是直接在数据点中随机抽取$k$个作为聚类初始中心点。不够好的初始值可能造成收敛速度很慢或者聚类失败。下面介绍两种改进方法。
- kmeans++方法，尽可能地使初始聚类中心分散开（spread-out）。
  这种方法首先随机产生一个初始聚类中心点，然后按照概率$P = \omega(x-c_i)^2$选取其他的聚类中心点。其中$\omega$是归一化系数。

- 多次初始化，保留最好的结果。

### K值的选取
在上面的讨论中，我们一直把$k$当做已经给定的参数。但是在实际操作中，聚类类别数通常都是未知的。如何确定超参数$k$呢？

我们可以使用不同的$k$值进行试验，并作出不同试验收敛后目标函数随$k$的变化。如下图所示，分别使用$1$到$4$之间的几个数字作为$k$值，曲线在$k=2$处有一个较为明显的弯曲点（elbow point）。故确定$k$取值为2。
![参数K的确定](/img/kmeans_object_fun_vs_k.png)

### 距离的度量
目标函数是各个cluster中的点与其中心点的距离均值。其中就涉及到了“距离”的量度。下面是几种较为常用的距离度量方法。
- 欧几里得距离（最为常用）
- 余弦距离（向量的夹角）
- 核函数（[Kernel K-Means](http://www.public.asu.edu/~jye02/CLASSES/Fall-2005/PAPERS/kdd_spectral_kernelkmeans.pdf)）

### 迭代终止条件
当判断算法已经收敛时，退出迭代。常用的迭代终止条件如下：
- 达到了预先给定的最大迭代次数
- 在将数据点assign到不同聚类中心时，和上轮结果没有变化（说明已经收敛）
- 目标函数（平均的距离）下降小于阈值

## 基于K-Means的图像分割
图像分割中可以使用K-Means算法。我们首先将一副图像转换为特征空间（Feature Space）。例如使用灰度或者RGB的值作为特征向量，就得到了1D或者3D的特征空间。之后，可以基于Feature Space上各点的相似度，将它们聚类。这样，就完成了对原始图像的分割操作。如下所示。
![图像分割结果1](/img/kmeans_image_seg_via_intensity.png)

然而，上述方法有一个缺点。那就是在真实图像中，属于同一个物体的像素点常常在空间上是相关的（Spatial Coherent）。而上述方法没有考虑到这一点。我们可以根据颜色亮度和空间上的距离构建Feature Space，获得更加合理的分割效果。

在2012年PAMI上有一篇文章[SLIC Superpixels Compared to State-of-the-art Superpixel Methods](https://infoscience.epfl.ch/record/177415/files/Superpixel_PAMI2011-2.pdf)介绍了使用K-Means进行超像素分割的相关内容，可以参考。这里不再赘述了。

## 优点和不足
作为非监督学习中的经典算法，K-Means的一大优点便是实现简单，速度较快，最小化条件方差（conditional variance，good represention of data，这里没有具体解释，不太懂）。

它的缺点主要有：
- 对outlier比较敏感。如下图所示。由于outlier的存在，右边的cluster硬是被拉长了。这方面，K-Medians更有鲁棒性。
![outlier是个大麻烦](/img/kmeans_sensitive_to_outlier.png)
- 每个点只能确定地属于或不属于某一个cluster。可以参考高斯混合模型和Fuzzy K-Means的soft assignment。
- 在round shape情况下性能比较好（想想圆形和方形各点的欧几里得距离）。而且每个cluster最好数据点数目和分布的密度相近。
- 如果数据点有非凸形状分布，算法表现糟糕。可以参考谱聚类（Spectral Clustering）或者kernel K-Means。

针对K-Means，也有不少相关改进工作，参考下面这幅图吧。
![K-Means Scaling Up](/img/kmeans_scaling_up.png)

## PS
这里插一句与本文内容完全无关的一个tip：在Markdown中使用LaTex时，如果在内联形式（也就是行内使用两个美元符号插入公式）时，如果有多个下划线，那么Markdown有时会将下划线之间的部分认为是自己的斜体标记，如下所示：
```
$c_{i}^\ast$ XXX $\delta_{ij}^\ast$
```
它的显示效果为$c_{i}^\ast$ XXX $\delta_{ij}^\ast$。

这时候，只需在下划线前加入反斜线进行转义即可（虽然略显麻烦）。如下所示：
```
$c\_{i}^\ast$ XXX $\delta\_{ij}^\ast$
```
它的显示效果为$c\_{i}^\ast$ XXX $\delta\_{ij}^\ast$。

具体分析可以参见[博客](http://lukang.me/2014/mathjax-for-hexo.html)。
