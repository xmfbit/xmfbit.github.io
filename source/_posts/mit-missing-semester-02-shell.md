---
title: MIT Missing Semester - Shell
date: 2020-03-13 22:05:30
tags:
    - bash
    - tool
---

工欲善其事，必先利其器。[MIT Missing Semester](https://missing.csail.mit.edu/)就是这样一门课。在这门课中，不会讲到多少理论知识，也不会告诉你如何写代码，而是会向你介绍诸如shell，git等常用工具的使用。这些工具其实自己在学习工作中或多或少都有接触，不过还是有一些点是漏掉的。所以，一起来和MIT的这些牛人们重新熟悉下这些工具吧！

这篇博客，包括后续的几篇，是我个人在过课程lecture的时候随手记下的自己之前不太清楚的点，可能并不适合阅读到这篇文章的你。如果有时间，还是建议去课程网站上自己过一遍。

这里我跳过了第一节课，直接从bash shell开始。

<!--more -->

## 一些零散的点

- bash中双引号和单引号的区别

双引号会发生变量替换，单引号不会。

``` bash
foo=bar

# output: hello, bar
echo "hello, $foo"
# output: hello, $foo
echo 'hello, $foo'
```

- bangbang

使用`!!`可以执行上一条命令。

- bash 中的特殊变量

``` bash
$? # 上一条命令的返回值，正常退出是0，否则是非0
$@ # 所有输入的参数
$# # 输入参数的个数
$$ # pid of current script

# 检查上条命令是否正常退出

if [ $? -ne 0 ]; then
  echo "fail"
else
  ehco "success"
fi
```

- 如何忽略命令的输出

有的时候，我们只想要命令的返回值。例如使用`grep foo bar`来查看文件`bar`中是否含有字符串`foo`，可以将标准输出和标准错误重定向到`/dev/null`。

``` bash
# 第一个是标准输出，第二个是标准错误
grep "foo" bar > /dev/null 2> /dev/null

# 或者可以这样：

# grep "name" test_lazy.cpp 2>&1 > /dev/null

if [ "$?" -ne 0 ]; then
    echo "found foo in bar"
else
    echo "not found foo in bar"
fi
```

## globbing

- 任意字符：`*`
- 单个字符：`?`
- 使用`{}`给定可选元素的集合。

``` bash
a.{hpp,cpp}  => a.hpp a.cpp
a{,0,1,2}   => a a0 a1 a2
# 支持多层级
touch proj{1,2}/{a,b}.txt
# 还支持range
touch proj{1,2}/{a..g}.txt
```

## bash 中的函数

- 如何写函数

``` bash
mycd () {
    cd $1
    pwd
}
```

- 如何在bash中导入脚本中的函数

``` bash
source your_bash_script.sh
# then use the function defined in the bash script
# 你可以这样理解：from your_bash_script import *
```

## for-loop

### 遍历给定的元素序列

使用`for item in xxx; do yyy; done`来遍历给定的序列，并施加具体操作于序列元素：

``` bash
for i in 1 2 3
do
  echo "welcome $i"
done
# welcome 1
# welcome 2
# welcome 3
```

注意，列表元素是通过空格来隔离的。如果这样写

``` bash
for i in 1, 2, 3
```

那么最终输出也是`welcome 1, welcome 2, welcome 3`

还可以使用for-range的方法：

``` bash
for i in {1..3}
```

### c-style for-loop

也可以像C语言那样使用for-loop：

``` bash
for (( Exp1; Exp2; Exp3)); do xxx; done
```

例如：

``` bash
for (( c=1; c<=3; c++ )); do echo "welcome $c"; done
```

还可以使用这种风格构造无穷循环，`for (( ; ; )); do xxx; done`。

### break / continue

当满足一定条件时，使用`break`退出循环，或使用`continue`继续循环。

### while

除了for-loop，还可以使用`while`。

``` bash
while CONDITION; do xxx; done
```

### until

`until`和`while`的用法一致，不同点在于：

- `while`是CONDITION为`true`执行，当`false`是退出循环
- `until`是CONDITION为`false`执行，当`true`时退出循环

``` bash
c=1
until [ $c -gt 3 ]; do
  echo "welcome $c"
  ((c++))
done
```

## 数学表达式

在上面for-loop中，已经看到了我们使用`((exp))`的形式进行数学表达式运算。一般来说，在bash shell中进行数学表达式的运算可以采用：

- 使用`expr`，如`c=$(expr 1 + 1); echo $c`，注意操作数与操作符之间都是有空格的。
- 使用`let`，如`c=1; let c=$c+1; echo $c`，注意操作数与操作符之间没有空格。
- 使用双括号`(())`，就像上面看到的那样：`c=1; echo $((c += 1)); echo $c`。这时候，操作数与操作符之间的空格可有可无。

最后一种双括号可能更为常用，支持的操作符：`+/-/++/--/*/%/**`，也支持逻辑运算符：`>=/<=/>/</==/!=/!/||/&&`。

如果希望进行浮点数运算，bash本身是不支持的，可以使用`bc`命令，将表达式作为字符串传入就可以了：

``` bash
echo "1.0+2.0" | bc
c=$(r=1.5;echo "$r + 2.5"|bc); echo $c
```

## 调试工具

shellcheck可以用来帮助静态分析shell脚本。用法：

``` bash
shellcheck your_bash_script.sh
```

可以去网站上试用：[Shellcheck](https://www.shellcheck.net/#)

## 几个有用的命令

这里列出一些常用的命令工具，都是和查找有关。更多内容，可以通过`man`或者`tldr`查看。

### 查找文件 find

最常用的用法：

``` bash
# 递归地查找当前目录及子目录下所有的python文件
find . -name="*.py"
# -type d 表示过滤结果为所有目录
# -type f 表示过滤结果为所有文件
find . -name="test" -type d
# Find all files modified in the last day
find . -mtime -1
# Find all zip files with size in range 500k to 10M
find . -size +500k -size -10M -name '*.tar.gz'
```

`find`还可以通过`-exec`来接后续处理，如：

``` bash
# Delete all files with .tmp extension
# 注意最后的 \
find . -name '*.tmp' -exec rm {} \;
# Find all PNG files and convert them to JPG
find . -name '*.png' -exec convert {} {.}.jpg \;
```

你也可以用`fd`作为`find`的改进版。具体用法可以参考[fd](https://github.com/sharkdp/fd)，这里不再多说了。

### locate

如果你想按照名字去查找文件，还可以试试`locate`。一个简单的比较：

- `locate`只支持按名字查找，`find`可以更加多样
- `locate`通过周期性更新的database来查找，时效性不如`find`
- `locate`使用更简单，默认会查找所有符合要求的文件，`find`一般是查找给定路径下的文件

由于上述原因，我一般是使用`locate`查找系统自带的某个lib等文件。比如有时候我可能不知道`libcudart.so`在哪里，这时候就可以通过`locate libcudart.so`来查找。

``` bash
locate libcudart.so | grep "/usr"

# output:
# /usr/local/cuda-10.0/doc/man/man7/libcudart.so.7
# /usr/local/cuda-10.0/lib64/libcudart.so
# ...
```

### 在文件中查找字符串 grep

`grep` 用来在文件中正则匹配字符串，比如某个变量或函数定义啥的。`grep`命令想当强大，在胡须课程中还会着重介绍。

``` bash
# 在文件中查找xxx，并打印其所在的行
grep xxx file

# 在所有文件中递归地查找
grep -R xxx .
```

常用的一些flag，可以是`-C +number`（用来显示match的context，number是行数），`-v`是反转（不包含所给pattern的行）

和`find`一样，`grep`也有一些更好用的替代品，如`rg`，`ag`，`ack`等。

``` bash
# Find all python files where I used the requests library
rg -t py 'import requests'
# Find all files (including hidden files) without a shebang line
rg -u --files-without-match "^#!"
# Find all matches of foo and print the following 5 lines
rg foo -A 5
# Print statistics of matches (# of matched lines and files )
rg --stats PATTERN
```

### 查找历史命令 history

`history`可以打印历史的shell命令，和`grep`配合能够找到历史上曾经用过的某给定命令。不过这个我在使用`zsh`的时候，一般是通过光标上下键来联想查找的。

一个有用的工具：`fzf`（which means 模糊查找）。

另外，这里讲师推荐了一款基于历史命令的自动补全（看lecture时候觉得很酷）。如果你和我一样使用`zsh`，可以参考这个插件：[zsh-autosuggestions](https://github.com/zsh-users/zsh-autosuggestions)。

### 关于目录

因为shell环境下没有GUI，所以查看一个目录内的内容，包括跳转目录都很不方便。对此也有一些好用的工具：

- 查看目录内容：`tree`（最经典），`broot`，`nnn`，`ranger`
- 跳转目录：`autojump`（在用），`fasd`

## 课后习题

### 关于ls的用法

- Includes all files, including hidden files：`ls -al`
- Sizes are listed in human readable format (e.g. 454M instead of 454279954)：`ls -lh`
- Files are ordered by recency：`ls -lt`
- Output is colorized：`ls -l --color` （zsh自动colorized，所以这个没有验证）

### bash函数

Write bash functions `marco` and `polo` that do the following. Whenever you execute `marco` the current working directory should be saved in some manner, then when you execute `polo`, no matter what directory you are in, `polo` should cd you back to the directory where you executed `marco`. For ease of debugging you can write the code in a file `marco.sh` and (re)load the definitions to your shell by executing `source marco.sh`.

``` bash
#!/bin/bash

# 使用文件存储要cd的path
macro () {
    echo "$(pwd)" > $HOME/.macro.history
}

polo () {
    cd "$(cat "$HOME/.macro.history")" || exit 1
}
```

### 循环和程序返回值判断

Say you have a command that fails rarely. In order to debug it you need to capture its output but it can be time consuming to get a failure run. Write a bash script that runs the following script until it fails and captures its standard output and error streams to files and prints everything at the end. Bonus points if you can also report how many runs it took for the script to fail.

``` bash
#!/bin/bash

for ((i=1; ; i++)); do
  # save the script to `fail_rarely.sh`
  ./fail_rarely.sh 2&> out.log
  if [ $? -ne 0 ]; then
    echo "fail after run $i times"
    echo "stdout and stderr message:"
    cat out.log
    break
  fi
done
```

### xargs和管道

As we covered in lecture find’s `-exec` can be very powerful for performing operations over the files we are searching for. However, what if we want to do something with all the files, like creating a zip file? As you have seen so far commands will take input from both arguments and STDIN. When piping commands, we are connecting STDOUT to STDIN, but some commands like tar take inputs from arguments. To bridge this disconnect there’s the `xargs` command which will execute a command using STDIN as arguments. For example `ls | xargs rm` will delete the files in the current directory.

Your task is to write a command that recursively finds all HTML files in the folder and makes a zip with them. Note that your command should work even if the files have spaces (hint: check `-d` flag for `xargs`)

#### xargs

先来看下`xargs`和管道的区别。这里已经给了一个例子：`ls | xargs rm`。由于`rm`命令比较危险，所以下面会换成`cat`（删除文件变成了打印文件内容）。

为什么不能用管道呢，比如`ls | cat`。我们先建立一个空目录作为playground：

``` bash
mkdir test
cd test
echo "simgple test" > a.txt
```

执行`ls | cat`，会发现它只是把当前目录下的所有文件名打印了出来，并没有打印`a.txt`的内容：

``` bash
ls | cat
# a.txt
```

这是因为管道只是把STDOUT作为`cat`的STDIN。在linux中，STDOUT和STDIN是两个特殊的文件，`ls`将把它的输出结果写入到STDOUT中，同时我们就会在屏幕上看到对应输出。而`cat`从STDIN中接受输入。当没有管道时，由用户输入并写入STDIN。由于管道，`cat`将直接从STDOUT中读取。也就是`ls`的输出，也就是当前目录下的文件列表。拆解后想当于下面：

``` bash
ls > stdout_ls
cat < stdout_ls
```

所以，如果我们想要打印`a.txt`的内容，管道就不够用了。也就是上面说的，我们要把`ls`的输出作为`cat`的参数。这时候需要使用`xargs`：

``` bash
ls | xargs cat
# simple test
```

#### 准备

先准备一些测试文件

``` bash
mkdir htmls
cd htmls
mkdir htmls/{1..3}
touch htmls/1/1.html
touch htmls/2/2\ 2.html
touch htmls/root.html
```

#### 实现

题目说明中的`-d`没找到，在[12 Practical Examples of Linux Xargs Command for Beginners](https://www.tecmint.com/xargs-command-examples/)找到了如下用法，使用
`-print0`和`-0`（是数字`0`）配合，具体可以参考`man xargs`中的内容。

``` bash
#-0      Change xargs to expect NUL (``\0'') characters as separators, instead of spaces and newlines.  
#        This is expected to be used in concert with the -print0 function in find(1).

find htmls -name "*.html" -print0 | xargs -0 tar vcf html.zip
```

### 命令组合 

Write a command or script to recursively find the most recently modified file in a directory. More generally, can you list all files by recency?

首先递归地列出当前目录下的所有文件，再使用`ls -lt`将其按照时间排序。

``` bash
find -L . -type f -print0 | xargs -0 ls -lt

# 如果只需要最新的那个，使用 head命令只打印第一行
find -L . -type f -print0 | xargs -0 ls -lt | head -1
```
