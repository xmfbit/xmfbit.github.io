---
title: Hack PyCaffe
date: 2018-03-16 19:01:32
tags:
    - caffe
    - python
---
这篇文章主要是[Github: PyCaffe Tutorial](https://github.com/nitnelave/pycaffe_tutorial/blob/master/04%20Hacking%20the%20Python%20API.ipynb)中Hack Pycaffe的翻译整理。后续可能会加上一些使用boost和C++为Python接口提供后端的解释。这里主要讨论如何为Pycaffe添加自己想要的功能。至于Pycaffe的使用，留待以后的文章整理。
![Python&&CPP binding](/img/caffe-hack-pycaffe-python-cpp-binding.jpg)
<!-- more -->

## PyCaffe的代码组织结构
见Caffe的`python`目录。下面这张图是与PyCaffe相关的代码的分布。其中`src`和`include`是Caffe框架的后端C++实现，`python`目录中是与PyCaffe关系更密切的代码。可以看到，除了`_caffe.cpp`以外，其他都是纯python代码。`_caffe.cpp`使用boost提供了C++与python的绑定，而其他python脚本在此层的抽象隔离之上，继续完善了相关功能，提供了更加丰富的API、
![代码组织结构](/img/hack-pycaffe-code-organization.png)

## 添加纯Python功能
首先，我们介绍如何在C++构建的PyCaffe隔离之上，用纯python实现想要的功能。

### 添加的功能和PyCaffe基本平行，不需要改变已有代码
有的时候想加入的功能和PyCaffe的关系基本是平行的，比如想仿照`PyTorch`等框架，加入对数据进行预处理的`Transformer`功能（这个API其实已经在PyCaffe中实现了，这里只是举个例子）。为了实现这个功能，我们可能需要使用`numpy`和`opencv`等包装图像的预处理操作，但是和Caffe本身基本没什么关系。在这样的情况下，我们直接编写即可。要注意在`python/caffe/__init__.py`中import相关的子模块或函数。这个例子可以参考`caffe.io`的实现（见`python/caffe/io.py`文件）。

### 添加的功能需要Caffe的支持，向已有的类中添加函数
如果添加的功能需要Caffe的支持，可以在`pycaffe.py`内添加，详见`Net`的例子。由于python的灵活性，我们可以参考`Net`的实现方式，待函数实现完成后，使用`<class>.<function> = my_function`动态地添加。如下所示，注意`_Net_forward`函数的第一个参数必须是`self`。

``` py
def _Net_forward(self, blobs=None, start=None, end=None, **kwargs):
    # do something
Net.forward = _Net_forward
```

与之相似，我们还可以为已经存在的类添加字段。注意，函数用`@property`装饰，且参数有且只有一个`self`，

``` py
# This function will be called when accessing net.blobs
@property
def _Net_blobs(self):
    """
    An OrderedDict (bottom to top, i.e., input to output) of network
    blobs indexed by name
    """
    if not hasattr(self, '_blobs_dict'):
        self._blobs_dict = OrderedDict(zip(self._blob_names, self._blobs))
    return self._blobs_dict 

# Set the field `blobs` to call _Net_blobs
Net.blobs = _Net_blobs
```

PyCaffe中已经实现的类主要有：`Net, SGDSolver, NesterovSolver, AdaGradSolver, RMSPropSolver, AdaDeltaSolver, AdamSolver`。

## 使用C++添加功能
当遇到如下情况时，可能需要修改C++代码：
- 为了获取更底层的权限控制，如一些私有字段。
- 性能考虑。

这时，你应该去修改`python/caffe/_caffe.cpp`文件。这个文件使用了boost实现了python与C++的绑定。

为了添加一个字段，可以在`Blob`部分添加如下的代码。这样，就会将python中`Blob`类的`num`字段绑定到C++的`Blob<Dtype>::num()`方法上。
``` cpp
.add_property("num", &Blob<Dtype>::num)
```

使用`.def`可以为python相应的类绑定方法。在下面的代码中，首先实现了`Net_Save`方法，然后将其绑定到了python中`Net`类的`save`方法上。这样，通过python调用`net.save(filename)`即可。

注意，当你修改了`_caffe,cpp`后，记得使用`make pycaffe`重新生成动态链接库。

``` cpp
# Declare the function
void Net_Save(const Net<Dtype>& net, string filename) {
    // ...
}

// ...

bp::class_<Net<Dtype>>("Net", bp::no_init)
# Now we can call net.save(file)
.def("save", &Net_Save)
```

当然，上面介绍的这些还很基础，关于boost的python绑定，可以参考官方的文档：[boost: python binding](http://www.boost.org/doc/libs/1_58_0/libs/python/doc/tutorial/doc/html/index.html)