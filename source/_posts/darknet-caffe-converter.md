---
title: YOLO Caffe模型转换BN的坑
date: 2019-03-09 12:51:18
tags:
     - caffe
     - deep learning
---
YOLO虽好，但是Darknet框架实在是小众，有必要在Inference阶段将其转换为其他框架，以便后续统一部署和管理。Caffe作为小巧灵活的老资格框架，使用灵活，方便魔改，所以尝试将Darknet训练的YOLO模型转换为Caffe。这里简单记录下YOLO V3 原始Darknet模型转换为Caffe模型过程中的一个坑。

<!-- more -->

# Darknet中BN的计算

以CPU代码为例，在Darknet中，BN做normalization的操作如下，[normalize_cpu](https://github.com/pjreddie/darknet/blob/master/src/blas.c#L147)

``` cpp
void normalize_cpu(float *x, float *mean, float *variance, int batch, int filters, int spatial)
{
    int b, f, i;
    for(b = 0; b < batch; ++b){
        for(f = 0; f < filters; ++f){
            for(i = 0; i < spatial; ++i){
                int index = b*filters*spatial + f*spatial + i;
                x[index] = (x[index] - mean[f])/(sqrt(variance[f]) + .000001f);
            }
        }
    }
}
```

可以看到，Darknet中的BN计算如下：
$$\hat{x} = \frac{x - \mu}{\sqrt{\sigma^2} + \epsilon}$$

而且，$\epsilon$参数是固定的，为$1\times 10^{-6}$。

# 问题和解决

然而，在Caffe（以及大部分其他框架）中，$\epsilon$的位置是在根号里面的，也就是：
$$\hat{x} = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}}$$

另外，查看`caffe.proto`可以知道，Caffe默认的$\epsilon$值为$1\times 10^{-5}$。

所以，在转换为caffe prototxt时，需要设置`batch_norm_param`如下：

``` proto
batch_norm_param {
  use_global_stats: true
  eps: 1e-06
}
```

另外，需要重新求解$\sigma^2$，按照layer输出要相等的等量关系，可以求得：

``` python
def convert_running_var(var, eps=DARKNET_EPS):
    return np.square(np.sqrt(var) + eps) - eps
```

这里调整之后，转换后的Caffe模型和原始Darknet模型的输出误差已经是$1\times 10^{-7}$量级，可以认为转换成功。
