---
title: VIM安装YouCompleteMe和Jedi进行自动补全
date: 2018-10-02 22:30:54
tags:
    - vim
    - tools
---
这篇主要记录自己尝试编译Anaconda + VIM并安装Jedi和YouCompleteMe自动补全插件的过程。踩了一些坑，不过最后还是装上了。给VIM装上了Dracula主题，有点小清新的感觉~

![我的VIM](/img/vim-config-demo.png)

<!-- more -->

## 使用Jedi和YouCompleteMe配置Vim
在远程开发机上调试代码时，我的习惯是大型项目使用sshfs将其镜像到本地，然后使用VSCode打开编辑。VSCode中有终端可以方便的ssh到远端开发机，我将"CTRL+`"配置成了编辑器和终端之间的切换快捷键。加上vim插件，就可以实现不用鼠标，不离开当前编辑环境进行代码编写和调试了。

然而，如果是想在开发机上写一段小的代码，上述方法就显得太麻烦了。

## 编译Vim
编译Vim，注意我们要设定其安装目录为anaconda下的bin目录：

``` sh
./configure --with-features=huge --enable-multibyte --enable-pythoninterp=yes --with-python-config-dir=/path/to/anaconda/bin/python-config --enable-gui=gtk2 --prefix=/path/to/anaconda
```

编译并安装：
``` sh
make -j4 VIMRUNTIMEDIR=/path/to/anaconda/share/vim/vim81
make install
```

安装后，可以查看vim的version进行确认。安装没有问题，会提示刚才编译的版本信息。
``` sh
vim --version
```

使用Vundle管理插件，这个没有什么问题，直接按照README提示即可，见：[Vundle@Github](https://github.com/VundleVim/Vundle.vim)。

使用Vundle进行插件管理，只需要以下面的形式指明插件目录或Github仓库名称，进入vim后，在Normal状态，输入`:PluginInstall`即可。

## Jedi
首先需要安装jedi的python包：
``` sh
pip install jedi
```

使用Vbudle安装[jedi-vim](https://github.com/davidhalter/jedi-vim)，并在`.vimrc`中添加以下内容。
```
let g:jedi#force_py_version=2.7
```

## YouCompleteMe
使用Vundle安装[YouCompleteMe](https://github.com/Valloric/YouCompleteMe#ubuntu-linux-x64)。

之后，进入目录`.vim/bundle/YouCompleteMe`，执行`./install.py`。如果需要C++支持，执行`./install.py --clang-completer`。

但是，其中遇到了问题，找不到Python.h文件。使用`locate Python.h`，明确该文件确实存在，且其位于`/path/to/anaconda/include/python2.7`后，手动修改CMakeLists.txt，指定该文件目录位置即可。

修改这个：
`.vim/bundle/YouCompleteMe/third_party/ycmd/cpp/CMakeLists.txt`
和
`.vim/bundle/YouCompleteMe/third_party/ycmd/cpp/ycm/CMakeLists.txt`，向其中添加：

``` sh
set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -I/path/to/anaconda/include/python2.7" )
```

强行指定头文件包含目录。

## 括号自动补全 
虽然SO上有人指出可以直接通过设置`.vimrc`的方法实现，不过还是直接用现成的插件吧。推荐使用[jiangmiao/auto-pairs](https://github.com/jiangmiao/auto-pairs)。可以按照README的说明进行安装。
