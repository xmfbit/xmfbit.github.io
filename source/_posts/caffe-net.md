---
title: Caffe中的Net实现
date: 2018-02-28 10:16:43
tags:
     - caffe
---
Caffe中使用`Net`实现神经网络，这篇文章对应Caffe代码总结`Net`的实现。
<img src="/img/caffe-net-demo.jpg" width = "300" height = "200" alt="Net示意" align=center />
<!-- more-->

## proto中定义的参数
```
message NetParameter {
  // net的名字
  optional string name = 1; // consider giving the network a name
  // 以下几个都是弃用的参数，为了定义输入的blob（大小）
  // 下面有使用推荐`InputParameter`进行输入设置的方法
  // DEPRECATED. See InputParameter. The input blobs to the network.
  repeated string input = 3;
  // DEPRECATED. See InputParameter. The shape of the input blobs.
  repeated BlobShape input_shape = 8;

  // 4D input dimensions -- deprecated.  Use "input_shape" instead.
  // If specified, for each input blob there should be four
  // values specifying the num, channels, height and width of the input blob.
  // Thus, there should be a total of (4 * #input) numbers.
  repeated int32 input_dim = 4;
  
  // Whether the network will force every layer to carry out backward operation.
  // If set False, then whether to carry out backward is determined
  // automatically according to the net structure and learning rates.
  optional bool force_backward = 5 [default = false];
  // The current "state" of the network, including the phase, level, and stage.
  // Some layers may be included/excluded depending on this state and the states
  // specified in the layers' include and exclude fields.
  optional NetState state = 6;

  // Print debugging information about results while running Net::Forward,
  // Net::Backward, and Net::Update.
  optional bool debug_info = 7 [default = false];

  // The layers that make up the net.  Each of their configurations, including
  // connectivity and behavior, is specified as a LayerParameter.
  repeated LayerParameter layer = 100;  // ID 100 so layers are printed last.

  // DEPRECATED: use 'layer' instead.
  repeated V1LayerParameter layers = 2;
}
```

### Input的定义
在`train`和`deploy`的时候，输入的定义常常是不同的。在`train`时，我们需要提供数据$x$和真实值$y$，这样网络的输出$\hat{y} = \mathcal{F}_\theta (x)$与真实值$y$计算损失，bp，更新网络参数$\theta$。


在`deploy`时，推荐使用`InputLayer`定义网络的输入，下面是`$CAFFE/models/bvlc_alexnet/deploy.prototxt`中的输入定义：
```
layer {
  name: "data"
  type: "Input"
  // 该层layer的输出blob名称为data，供后续layer使用
  top: "data"
  // 定义输入blob的大小：10 x 3 x 227 x 227
  // 说明batch size = 10
  // 输入彩色图像，channel = 3, RGB
  // 输入image的大小：227 x 227
  input_param { shape: { dim: 10 dim: 3 dim: 227 dim: 227 } }
}
```

## 头文件
`Net`的描述头文件位于`$CAFFE/include/caffe/net.hpp`中。
