---
title: 在Caffe中使用Baidu warpctc实现CTC Loss的计算
date: 2017-02-22 15:34:32
tags:
     - caffe
---
CTC(Connectionist Temporal Classification) Loss 函数多用于序列有监督学习，优点是不需要对齐输入数据及标签。本文内容并不涉及CTC Loss的原理介绍，而是关于如何在Caffe中移植Baidu美研院实现的[warp-ctc](https://github.com/baidu-research/warp-ctc)，并利用其实现一个LSTM + CTC Loss的验证码识别demo。下面这张图引用自warp-ctc的[项目页面](https://github.com/baidu-research/warp-ctc)。本文介绍内容的相关代码可以参见我的GitHub项目[warpctc-caffe](https://github.com/xmfbit/warpctc-caffe)
![CTC Loss](/img/warpctc_intro.png)
<!-- more -->

## 移植warp-ctc
本节介绍了如何将`warp-ctc`的源码在Caffe中进行编译。

首先，我们将`warp-ctc`的项目代码从GitHub上clone下来。在Caffe的`include/caffe`和`src/caffe`下分别创建名为`3rdparty`的文件夹，将warp-ctc中的头文件和实现文件分别放到对应的文件夹下。之后，我们需要对其代码和配置进行修改，才能在Caffe中顺利编译。

由于`warp-ctc`中使用了`C++11`的相关技术，所以需要修改Caffe的`Makefile`文件，添加`C++11`支持，可以参见[Makefile](https://github.com/xmfbit/warpctc-caffe/blob/master/Makefile)。

对Caffe的修改就是这么简单，之后我们需要修改`warp-ctc`中的代码文件。这里的修改多且乱，边改边试着编译，所以可能其中也有不必要的修改。最后的目的就是能够使用GPU进行CTC Loss的计算。

`warp-ctc`提供了CPU多线程的计算，这里我直接将相应的`openmp`并行化语句删掉了。

另外，需要将warp-ctc中包含CUDA代码的相关头文件后缀名改为`cuh`，这样才能够通过编译。否则编译器会给出找不到`__host__`和`__device__`等等关键字的错误。

对于详细的修改配置，还请参见GitHub相应的[代码文件](https://github.com/xmfbit/warpctc-caffe/blob/master/include/caffe/3rdparty/detail/hostdevice.cuh)。

## 实现CTC Loss计算
编译没有问题后，我们可以编写`ctc_loss_layer`实现CTC Loss的计算。在实现时，注意参考文件`ctc.h`。这个文件中给出了使用`warp-ctc`进行CTC Loss计算的全部API接口。

`ctc_loss_layer`继承自`loss_layer`，主要是前向和反向计算的实现。由于`warp-ctc`中只对单精度浮点数`float`进行支持，所以，对于双精度网络参数，直接将其设置为`NOT_IMPLEMENTED`，如下所示。

``` cpp
template <>
void CtcLossLayer<double>::Forward_cpu(
    const vector<Blob<double>*>& bottom, const vector<Blob<double>*>& top) {
    NOT_IMPLEMENTED;
}

template <>
void CtcLossLayer<double>::Backward_cpu(const vector<Blob<double>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<double>*>& bottom) {
    NOT_IMPLEMENTED;
}
```

使用`warp-ctc`相关接口进行CTC Loss计算的步骤如下：

- 设置`ctcOptions`，指定使用CPU或GPU进行计算，并指定CPU计算的线程数和GPU计算的CUDA stream。
- 调用`get_workspace_size()`函数，预先为计算分配空间（分配的内存根据计算平台不同位于CPU或GPU）。
- 调用`compute_ctc_loss()`函数，计算`loss`和`gradient`。

其中，在第三步中计算`gradient`时，可以直接将对应`blob`的`cpu/gpu_diff`指针传入，作为`gradient`。

这部分的实现代码分别位于`include/caffe/layers`和`src/caffe/layers/`下。

## 验证码数字识别
本部分相关代码位于`examples/warpctc`文件夹下。实验方案如下。

- 使用`Python`中的`capycha`进行包含`0-9`数字的验证码图片的产生，图片中数字个数从`1`到`MAX_LEN`不等。
- 使用`10`作为`blank_label`，将所有的标签序列在后面补`blank_label`以达到同样的长度`MAX_LEN`。
- 将图像的每一列看做一个time step，网络模型使用`image data->2LSTM->fc->CTC Loss`，简单粗暴。
- 模型训练过程中，数据输入使用`HDF5`格式。

### 数据产生
使用`captcha`生成验证码图片。[这里](https://pypi.python.org/pypi/captcha/0.1.1)是一个简单的API demo。默认生成的图片大小为`160x60`。我们将其长宽缩小一半，使用`80x30`的彩色图片作为输入。

使用`python`中的`h5py`模块生成`HDF5`格式的数据文件。将全部图片分为两部分，80%作为训练集，20%作为验证集。

### LSTM的输入
在Caffe中已经有了`lstm_layer`的实现。`lstm_layer`要求输入的序列`blob`为`TxNx...`，也就是说我们需要
### 训练

### 结论
