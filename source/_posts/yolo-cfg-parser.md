---
title: YOLO网络参数的解析与存储
date: 2017-03-06 15:51:22
tags:
     - yolo
     - deep learning
---
YOLO的原作者使用了自己开发的Darknet框架，而没有选取当前流行的其他深度学习框架实现算法，所以有必要对其网络模型的参数解析与存储方式做一了解，方便阅读源码和在其他流行的框架下的算法移植。
<!-- more -->

## YOLO网络结构定义的CFG文件
YOLO中的网络定义采用和Caffe类似的方式，都是通过一个顺序堆叠layer来对神经网络结构进行定义的文件来描述。不同的地方在于，Caffe中使用了Google家出品的protobuf，省时省力，无需自己实现解析文件的功能，但是也使得Caffe对第三方库的依赖更加严重。相信很多人在编译Caffe的时候都出现过无法链接等蛋疼无比的问题。而YOLO的作者则是使用了自己定义的一种CFG文件格式，需要自己实现解析功能。

CFG文件的格式可以归纳如下（可以打开某个[CFG文件](https://github.com/pjreddie/darknet/blob/master/cfg/yolo-voc.cfg)进行对照）：
```
[net]
# 这里会对net的参数进行配置
# 同时YOLO将对net的求解器的参数也放在了这里
[conv]
# 一些conv层的参数描述
[maxpool]
# 一些池化层的参数描述

# 顺序堆叠的其他layer描述
```
在Darknet的代码中，将每个`[]`符号导引的参数列表叫做section。

## 网络结构解析器 Parser
具体的解析实现参见[parser.c文件](https://github.com/pjreddie/darknet/blob/master/src/parser.c)。我们先以`convolutional_layer parse_convolutional(list *options, size_params params)`函数为例，看一下Darknet是如何完成对卷积层参数的解析的。

从函数签名可以看出，这个函数接受一个`list`的变量（Darknet中将堆叠起来的这些层描述抽象成链表），而`size_params`类型的变量`params`指示了该层上一层的参数情况，其具体定义如下：
``` cpp
typedef struct size_params{
    int batch;
    int inputs;
    int h;
    int w;
    int c;
    int index;
    int time_steps;
    network net;
} size_params;
```
这样，在构建该层卷积层的时候，我们就能够知道上一层的输入维度等信息，方便做一些参数检查和layer初始化等的工作。

进入函数内部，会发现频繁出现`option_find_int`这个函数。从函数名字面意义看，应该是要解析字符串中的整型数。

我们首先来看一下这个函数的定义吧~这个函数并不在`parser.c`中，而是在[option_list.c 文件](https://github.com/pjreddie/darknet/blob/master/src/option_list.c)中。

``` cpp
// l: data pointer to the list
// key: the key to find, example: "filters", "padding"
// def: default value
int option_find_int(list *l, char *key, int def)
{
    // 去找到该key对应的数值，使用atoi转换为整型数
    char *v = option_find(l, key);
    if(v) return atoi(v);
    // 使用XXX_quiet版本可以不打印此信息
    fprintf(stderr, "%s: Using default '%d'\n", key, def);
    // 没有找到key，返回默认值
    return def;
}
```

而其中的`option_find`函数则是逐项顺序查找，匹配字符串来实现的。

``` cpp
char *option_find(list *l, char *key)
{
    node *n = l->front;
    while(n){
        kvp *p = (kvp *)n->val;
        if(strcmp(p->key, key) == 0){
            p->used = 1;
            return p->val;
        }
        n = n->next;
    }
    return 0;
}
```

## 构建conv层
由此，我们可以通过CFG文件得到卷积层的参数了。接下来需要调用其初始化函数，进行构建。

``` cpp
    // 首先得到参数
    int n = option_find_int(options, "filters",1);
    int size = option_find_int(options, "size",1);
    int stride = option_find_int(options, "stride",1);
    int pad = option_find_int_quiet(options, "pad",0);
    int padding = option_find_int_quiet(options, "padding",0);
    if(pad) padding = size/2;
	// 激活函数是通过匹配其名称的方法得到的
    char *activation_s = option_find_str(options, "activation", "logistic");
    ACTIVATION activation = get_activation(activation_s);
    // 通过上层的信息得到batch size，做参数检查
    int batch,h,w,c;
    h = params.h;
    w = params.w;
    c = params.c;
    batch=params.batch;
    if(!(h && w && c)) error("Layer before convolutional layer must output image.");
    int batch_normalize = option_find_int_quiet(options, "batch_normalize", 0);
    int binary = option_find_int_quiet(options, "binary", 0);
    int xnor = option_find_int_quiet(options, "xnor", 0);
    // 调用初始化函数
    convolutional_layer layer = make_convolutional_layer(batch,h,w,c,n,size,stride,padding,activation, batch_normalize, binary, xnor, params.net.adam);
    layer.flipped = option_find_int_quiet(options, "flipped", 0);
    layer.dot = option_find_float_quiet(options, "dot", 0);
```

所以，如果在阅读源码时候，对layer的某个成员变量不知道什么意思的话，可以参考此文件，看一下原始解析对应的字符串是什么，一般这个字符串描述是比较具体的。

## 构建网络
有了各个layer的解析方法，接下来就可以逐层读取参数信息并构建网络了。

Darknet中对应的函数为`network parse_network_cfg(char *filename)`，这个函数接受文件名为参数，进行网络结构的解析。

首先，调用`read_cfg(filename)`得到CFG文件的一个层次链表，接着只要对这个链表进行解析就好了。不过对第一个section，也就是`[net]` section，要特殊对待。这里不再多说了。

## 保存参数信息
Darknet中保存带参数的layer的信息是直接写入二进制文件。仍然以卷积层为例，其保存代码如下所示：

``` cpp
void save_convolutional_weights(layer l, FILE *fp)
{
    if(l.binary){
        //save_convolutional_weights_binary(l, fp);
        //return;
    }
#ifdef GPU
    if(gpu_index >= 0){
        pull_convolutional_layer(l);
    }
#endif
    int num = l.n*l.c*l.size*l.size;
    fwrite(l.biases, sizeof(float), l.n, fp);
    // 由于darknet设计时，没有单独设计BN层，所以BN的参数也是和其所在的层一起保存的，如果读取时候要注意分别讨论
    if (l.batch_normalize){
        fwrite(l.scales, sizeof(float), l.n, fp);
        fwrite(l.rolling_mean, sizeof(float), l.n, fp);
        fwrite(l.rolling_variance, sizeof(float), l.n, fp);
    }
    fwrite(l.weights, sizeof(float), num, fp);
    if(l.adam){
        fwrite(l.m, sizeof(float), num, fp);
        fwrite(l.v, sizeof(float), num, fp);
    }
}
```

在保存整个网络的参数信息的时候，同样逐层保存到同一个二进制文件中就好了。
