---
title: 捉bug记 - JupyterNotebook中使用pycaffe加载多个模型一直等待的现象
date: 2018-02-27 13:30:25
tags:
    - caffe
---
JupyteNotebook是个很好的工具，但是在使用pycaffe试图在notebook中同时加载多个caffemodel模型的时候，却出现了无法加载的问题。

## bug重现
我想在notebook中比较两个使用不同方法训练出来的模型，它们使用了同样的LMDB文件进行训练。加载第一个模型没有问题，但当加载第二个模型时，却一直等待。在StackOverflow上我发现了类似的问题，可以见：[Can't load 2 models in pycaffe](https://stackoverflow.com/questions/37260158/cant-load-2-models-in-pycaffe)。

## 解决方法
这是由于pycaffe（是否要加上jupyter-notebook？因为不用notebook，以前没有出现过类似问题）不能并发读取同样的LMDB所导致的。但是很遗憾，没有发现太好的解决办法。最后只能是将LMDB重新copy了一份，并修改prototxt文件，使得两个模型分别读取不同的LMDB。