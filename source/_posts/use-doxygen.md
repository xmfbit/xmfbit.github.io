---
title: Windows环境下使用Doxygen生成注释文档
date: 2016-12-16 19:00:00
tags:
    - tool
    - doxygen
---

Doxygen 是一种很好用的代码注释生成工具，然而和很多国外的工具软件一样，在中文环境下，它的使用总是会出现一些问题，也就是中文注释文档出现乱码。经过调试，终于是解决了这个问题。

<!-- more -->
## 安装 Doxygen

Doxygen 在Windows平台下的安装比较简单，[Doxygen的项目主页](http://www.doxygen.nl/)提供了下载和安装的使用说明，可以下载它们的官方使用手册进行阅读。对于Windows，提供了源代码编译安装和直接安装程序安装两种方式，可以自行选择。

安装成功后，使用命令行命令

``` bash
doxygen --help
```

就可以查看帮助文档，对应参数含义一目了然，降低了入手难度。

使用命令，


``` bash
doxygen -g doxygen_filename
```

就可以在当前目录下建立一个doxygen配置文件，用文本编辑器打开就可以编辑里面的配置选项。

使用命令，

``` bash
doxygen doxygen_filename
```

就可以生成注释文档了。

下面就来说一说对中文的支持。

## 生成 HTML 格式文档

中文之所以乱码，很多时候是由于编码和译码格式不同，所以我们需要先知道自己代码文件的编码方式。我的代码都是建立在Visual Studio上的，可以通过VS的高级保存选项查看自己代码文件的存储编码格式。对于中文版的VS，一般应该是GB2312。

我们打开 Doxygen 的配置文件，将里面的 INPUT_ENCODING 改为我们代码文件的编码格式，这里就改成 GB2312。

这样一来，编译出来的 HTML 页面就不会有中文乱码了。

## 生成Latex 格式文档

生成 Latex 需要本机上安装有 Latex 的编译环境。如果是中文用户，推荐的是CTEX套件，可以到他们的网站上去下载。

可以看到，Doxygen为Latex文件的编译生成了make文件，我们在命令行窗口中执行make命令就可以完成编译，然而这时候会发现编译出错，pdf文档无法生成。

打开生成的refman.latex文档，添加宏包 CJKutf8。然后找到 `\begin{document}`一行，将其改为

``` latex
\begin{document}
\begin{CJK}{UTF8}{gbsn}
```

也就是说为正文提供了CJK环境，这样中文文本就可以正常编译了。

相应的，我们要将结尾的 `\end{document)`改为：
``` latex
\end{CJK}
\end{document}
```

这样，运行make命令之后，就可以看到中文的注释文档了。
