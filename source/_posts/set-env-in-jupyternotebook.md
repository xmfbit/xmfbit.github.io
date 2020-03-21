---
title: JupyterNotebook设置Python环境
date: 2018-04-09 13:44:04
tags:
    - python
    - tool
---
使用Python时，常遇到的一个问题就是Python和库的版本不同。Anaconda的env算是解决这个问题的一个好用的方法。但是，在使用Jupyter Notebook的时候，我却发现加载的仍然是默认的Python Kernel。这篇博客记录了如何在Jupyter Notebook中也能够设置相应的虚拟环境。
<!-- more -->
## conda的虚拟环境
在Anaconda中，我们可以使用`conda create -n your_env_name python=your_python_version`的方法创建虚拟环境，并使用`source activate your_env_name`方式激活该虚拟环境，并在其中安装与默认（主）python环境不同的软件包等。

当激活该虚拟环境时，ipython下是可以正常加载的。但是打开Jupyter Notebook，会发现其加载的仍然是默认的Python kernel，而我们需要在notebook中也能使用新添加的虚拟环境。

## 解决方法
解决方法见这个帖子：[Conda environments not showing up in Jupyter Notebook](https://stackoverflow.com/questions/39604271/conda-environments-not-showing-up-in-jupyter-notebook).

首先，安装`nb_conda_kernels`包：
``` bash
conda install nb_conda_kernels
```

然后，打开Notebook，点击`New`，会出现当前所有安装的虚拟环境以供选择，如下所示。
![选择特定的kernel加载](/img/set-env-in-notebook-choose-kernel.png)

如果是已经编辑过的notebook，只需要打开该笔记本，在菜单栏中选择`Kernel -> choose kernel -> your env kernel`即可。
![改变当前notebook的kernel](/img/set-env-in-notebook-change-kernel.png)

关于`nb_conda_kernels`的详细信息，可以参考其GitHub页面：[nb_conda_kernels](https://github.com/Anaconda-Platform/nb_conda_kernels)。