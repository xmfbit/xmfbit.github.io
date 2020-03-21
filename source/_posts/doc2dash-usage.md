---
title: doc2dash——制作自己的dash文档
date: 2017-08-26 19:32:00
tags:
    - tool
---
Dash是Mac上一款超棒的应用，提供了诸如C/C++/Python甚至是OpenCV/Vim等软件包或工具软件的参考文档。只要使用App的“Download Docsets”功能，就能轻松下载相应文档。使用的时候只需在Dash的搜索框内输入相应关键词，Dash会在所有文档中进行搜索，给出相关内容的列表。点击我们要寻找的条款，就能够直接在本地阅读文档。在Ubuntu/Windows平台上，Dash也有对应的替代品，例如[zeal](https://zealdocs.org)就是一款Windows/Linux平台通用的Dash替代软件。

这样强大的软件，如果只能使用官方提供的文档岂不是有些大材小用？[doc2dash](https://doc2dash.readthedocs.io/en/stable/)就是一款能够自动生成Dash兼容docset文件的工具。例如，可以使用它为PyTorch生成本地化的docset文件，并导入Dash/zeal中，在本地进行搜索阅读。这不是美滋滋？

本文章是基于doc2dash的官方介绍，对其使用进行的总结。
![Demo](/img/doc2dash_pytorch_example.jpg)

<!-- more -->
## 安装doc2dash
doc2dash是基于Python开发的。按照官方网站介绍，为了避免Python包的冲突，最好使用虚拟环境进行安装。我的机器上安装有Anaconda环境，所以首先使用`conda create`命令新建用于doc2dash的虚拟环境。
``` sh
conda create -n doc2dash
```
接下来，激活虚拟环境，并使用`pip install`命令安装。
``` sh
source activate doc2dash
pip install doc2dash
```

doc2dash支持的输出格式可以通过sphinx或者pydoctor。其中前者更加常用。下面以PyTorch项目的文档生成为例，介绍doc2dash的具体用法。
## 生成PyTorch文档
doc2dash使用sphinx生成相应的文档。在上述安装doc2dash的过程中，应该已经安装了sphinx包。不过我们还需要手动安装，以便处理rst文档。

```
pip install sphinx_rtd_theme
```
进入PyTorch的文档目录`docs/`，PyTorch已经为我们提供了Makefile，调用sphinx包进行文档处理，可以选择`make html`命令生成相应的HTML文档，生成的位置为`build/html`。

```
# in directory $PYTORCH/docs, run
make html
```

接下来，就可以使用doc2dash来继续sphinx的工作，生成Dash可用的文档文件了~使用`-n`指定生成的文件名称，后面跟source文件夹路径即可。

``` sh
# $PYTORCH/docs/build/html即为生成的HTML目录
doc2dash -n pytorch $PYTORCH/docs/build/html
```

之后，把生成的`pytorch.docset`导入到Dash中即可。如下图所示，点击“+”找到文件添加即可。
![添加docset](/img/doc2dash_how_to_add_docset.jpg)

## 在Ubuntu上安装zeal
zeal是Dash在非Mac平台上的替代软件。在Ubuntu上可以使用如下方式轻松安装（见[官方网站介绍](https://zealdocs.org/download.html#linux)）。

``` sh
sudo add-apt-repository ppa:zeal-developers/ppa
sudo apt-get update
sudo apt-get install zeal
```

安装后，可以使用`Tool/Docsets`下载相应的公开文档。如果想要添加自己生成的文档，只需要将生成的docset文件放到软件的文档库中即可，默认位置应在`$HOME/.local/share/Zeal/Zeal/docsets`。
