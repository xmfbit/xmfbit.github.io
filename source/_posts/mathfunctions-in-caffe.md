---
title: Caffe中的底层数学计算函数
date: 2017-03-08 17:24:48
tags:
---
Caffe中使用了BLAS库作为底层矩阵运算的实现，这篇文章对[mathfunction.hpp 文件](https://github.com/BVLC/caffe/blob/master/include/caffe/util/math_functions.hpp)中的相关函数做一总结。我们在自己实现layer运算的时候，也要注意是否Caffe中已经支持了类似运算，不要从scratch开始编码，自己累点不算啥，CPU/GPU的运算能力发挥不出来，更别说自己写的代码也许还是错的，那就尴尬了。。。

<!-- more -->
## BLAS介绍
以下内容参考[BLAS wiki页面](https://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms)整理。这里不涉及BLAS的过多内容，只为介绍Caffe中的相关函数做一过渡。

BLAS的全称是基础线性代数子程序库（Basic Linear Algebra Subprograms），提供了一些低层次的通用线性代数运算的实现函数，如向量的相加，数乘，点积和矩阵相乘等。BLAS的实现根绝硬件平台的不同而不同，常常利用了特定处理器的硬件特点进行加速计算（例如处理器上的向量寄存器和SIMD指令集），提供了C和Fortran语言支持。

不同的厂商根据自己硬件的特点，在BLAS的统一框架下，开发了自己的加速库，比如~~AMD的ACML~~（已经不再支持），Intel的MKL，ATLAS和OpenBLAS。其中后面的三个均可以在Caffe中配置使用。

在BLAS中，实现了矩阵与矩阵相乘的函数`gemm`（GEMM: General Matrix to Matrix Multiplication）和矩阵和向量相乘的函数`gemv`，这两个数学运算的高效实现，关系到整个DL 框架的运算速度。下面这张图来源于Jia Yangqing的博士论文。
![前向计算中的典型时间分布](/img/mathfunctions_time_distribution.png)

可以看到，在前向计算过程中，无论是CPU还是GPU，大量时间都花在了卷积层和全连接层上。全连接层不必多说，就是一个输入feature和权重的矩阵乘法。卷积运算也是通过矩阵相乘实现的。因为我们可以把卷积核变成一列，和相应的feature区域做相乘（如下图，这部分可以看一下Caffe中im2col部分的介绍和代码）。
![im2col的原理](/img/mathfunctions_im2col.png)

对于BLAS和GEMM等对DL的作用意义，可以参见这篇文章[Why GEMM is at the heart of deep learning](https://petewarden.com/2015/04/20/why-gemm-is-at-the-heart-of-deep-learning/)的分析。上面的图也都来源于这篇博客。
