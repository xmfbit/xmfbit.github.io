---
title: Ubuntu/Mac 工具软件列表
date: 2017-10-22 20:12:09
tags:
---
工作环境大部分在Ubuntu和MacOS上进行，这里是一个我觉得在这两个平台上很有用的工具的整理列表，大部分都可以在两个系统上找到。这里并不会给出具体安装方式，因为最好的文档总是软件的官方Document或者GitHub的README。
<!-- more -->

## zsh和Oh-my-zsh

如果经常在终端敲命令而且还在用系统自带的Bash？可以考虑试一下zsh替代bash，并使用[oh-my-zsh](https://github.com/robbyrussell/oh-my-zsh)武装zsh。

关于oh-my-zsh的帖子网上已经有很多，不过我还并没有用到太多的功能。oh-my-zsh中可以配置插件，不过我只是使用了`colored-man-pages`。顾名思义，它可以将使用`man`查询时的页面彩色输出。如下所示。
![彩色的cp man页面](/img/useful_tools_colored_man_pages.jpg)

## autojump

使用[autojump](https://github.com/wting/autojump)，可以很方便地在已经访问过的文件夹间快速跳转。甚至都不需要输入目标文件夹的全名，支持自动联想。

除了自动跳转功能，我还将其作为终端到文件资源管理器(Mac: Finder)的跳转功能。
```
# 跳转到path并使用文件资源管理器打开
jo path
```

## tldr

[tldr](https://github.com/tldr-pages/tldr) (too long don't read)是一款能够给出bash命令常用功能的工具。在Linux系统中，很多命令都有一长串参数。这其中很多都是不常用的。而我们使用时，常常是使用某几个常见的功能选项。tldr就能够给出命令的简要描述和例子。

例如，使用其查询`tar`的常用方法：

``` bash
tldr tar
# output
tar

Archiving utility.
Often combined with a compression method, such as gzip or bzip.

- Create an archive from files:
    tar cf target.tar file1 file2 file3

- Create a gzipped archive:
    tar czf target.tar.gz file1 file2 file3

- Extract an archive in a target folder:
    tar xf source.tar -C folder

- Extract a gzipped archive in the current directory:
    tar xzf source.tar.gz

- Extract a bzipped archive in the current directory:
    tar xjf source.tar.bz2

- Create a compressed archive, using archive suffix to determine the compression program:
    tar caf target.tar.xz file1 file2 file3

- List the contents of a tar file:
    tar tvf source.tar
```

tldr支持多种语言，我使用了python包安装。但是不知为何，tldr在我这里总显示奇怪的背景颜色，看上去很别扭。所以我实际使用的是[tldr-py](https://github.com/lord63/tldr.py)。

## tmux

用SSH登录到服务器上时，如果网络连接不稳定或是自己的主机意外断电，会造成正在跑的代码死掉。因为进程是依附于SSH的会话Session的。tmux是一个终端的“分线器”，可以很方便地将正在进行的终端会话detach掉，使其转入后台运行。正是有这一特点，所以我们可以在SSH会话时，新建tmux会话，在其中跑一些耗时很长的代码，而不必担心SSH掉线。当然，也可以将tmux作为一款终端多任务的管理软件，方便地在多个任务中进行跳转。不过这个功能，我更加常用的是下面的guake。

虽然Ubuntu14.04可以通过`apt-get`的方式安装tmux，不过为了能够使用一款好用的配置[oh-my-tmux](https://github.com/gpakosz/.tmux)（要求tmux>=2.1），还是推荐去GitHub上自己编译安装[tmux](https://github.com/tmux/tmux)。

## guake

[guake](https://github.com/Guake/guake)是一款Ubuntu上可以方便呼出终端的应用（按下F12，终端将以全屏的方式铺满桌面，F11可以切换全屏或半屏）。

## Dash/Zeal

Dash是Mac上一款用于查询API文档的软件。在Ubuntu或Windows上，我们可以使用Zeal这个替代软件。Zeal和Dash基本上无缝衔接，但是却是免费的（Mac上的软件真是好贵。。。）。之前我已经写过一篇[博客](https://xmfbit.github.io/2017/08/26/doc2dash-usage/)，介绍如何自己制作文档导入Zeal中。

## sshfs

使用sshfs可以在本地机器上挂载远程服务器某个文件夹。这样，操作本地的该文件夹就相当于操作远程服务器上的该文件夹（小心使用`rm`）。

## Alfred/Mutate

Alfred是Mac上一款非常好用的软件，就像蝙蝠侠身边的老管家一样，可以帮你自动化处理很多事情。除了原生
功能，还可以自己编写脚本实现扩展。例如查询豆瓣电影，查询ip，计算器等。鉴于这款软件的大名，这里不再多说。

[Mutate](https://github.com/qdore/Mutate)是Ubuntu上的一款替代软件。同时，它也提供了方便的扩展接口，只需要按照模板编写python/shell代码，可以很方便地将自己的自动化处理功能加入软件中。
