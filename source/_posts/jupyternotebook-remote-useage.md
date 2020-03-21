---
title: 远程登录Jupyter笔记本
date: 2017-02-26 19:53:11
tags:
    - tool
---
Jupyter Notebook可以很方便地记录代码和内容，很适合边写笔记边写示例代码进行学习总结。在本机使用时，只需在相应文件夹下使用`jupyter notebook`命令即可在浏览器中打开笔记页面，进行编辑。而本篇文章记述了如何在远端登录并使用Jupyter笔记本。这样，就可以利用服务器较强的运算能力来搞事情了。
![jupyternotebook](/img/jupyternotebook_logo.png)
<!-- more -->

## 配置jupter notebook
登录远程服务器后，使用如下命令生成配置文件。

``` bash
jupyter notebook --generate-config
```

并对其内容进行修改。打开配置文件：

``` bash
vim ~/.jupyter/jupyter_notebook_config.py
```

主要修改两处地方：

- `c.NotebookApp.ip='*'`，即不限制ip访问
- `c.NotebookApp.password = u'hash_value'`

上面的`hash_value`是由用户给定的密码生成的。可以使用`ipython`中的命令轻松搞定。

``` python
from notebook.auth import passwd
passwd()
"""
这里会要求用户输入密码并确认，生成的hash值要填写到上面
"""
```

当在服务器上运行jupyter-notebook时，我们常常不需要服务器上额外启动浏览器窗口。可以修改上述的配置文件，禁用服务器端的浏览器。找到下面这一行，改成`False`即可。

``` bash
c.NotebookApp.open_browser = True
```

## 启动notebook
之后，在远程服务器上启动笔记本`jupyter notebook`。接着，在本地机器上访问`远程服务器ip:8888`（默认端口为`8888`，也可以在配置文件中修改），输入密码即可访问远程笔记本。

## Jupyter Notebook的常用快捷键
类似Vim编辑器，Jupyter Notebook有两种键盘输入模式：
- 编辑模式，在单元中键入代码或文本，这时的单元框线是绿色的。通过`Esc`可以切换至命令模式。
- 命令模式，键盘输入运行命令，这时的单元框线是蓝色的。通过`Enter`可以切换至编辑模式。

常用的快捷键有：

- h：弹出快捷键列表
- m：将该Cell转换为Markdown输入状态
- y：将该Cell转换为代码输入状态
- shift+Enter：执行该Cell并将焦点置于下一个Cell（如果没有，将新建）
- Ctrl+Enter：执行该Cell


本篇内容参考自博客[远程访问jupyter notebook](http://blog.leanote.com/post/jevonswang/远程访问jupyter-notebook)以及[Jupyter Notebook快捷键
](http://www.cnblogs.com/weidiao/p/7792885.html)。
