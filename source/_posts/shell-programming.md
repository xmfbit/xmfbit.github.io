---
title: shell编程
date: 2017-11-10 13:06:30
tags:
    - linux
    - shell
---
介绍基本的shell编程方法，参考的教程是[Linux Shell Scripting Tutorial, A Beginner's handbook](http://www.freeos.com/guides/lsst/)。
![Bash Logo](/img/shell-programming-bash-logo.png)
<!-- more -->

## 变量

变量是代码的基本组成元素。可以认为shell中的变量类型都是字符串。

shell中的变量可以分为两类：系统变量和用户自定义变量。下面分别进行介绍。

在代码中使用变量值的时候，需要在前面加上`$`。`echo`命令可以在控制台打印相应输出。所以使用`echo $var`就可以输出变量`var`的值。

### 系统变量

系统变量是指Linux中自带的一些变量。例如`HOME`,`PATH`等。其中`PATH`又叫环境变量。更多的系统变量见下表：
![系统变量列表](/img/shell-programming-system-variables.jpg)

### 用户定义的变量

用户自定义变量是用户命名并赋值的变量。使用下面的方法定义：

``` bash
# 注意不要在等号两边插入空格
name=value
# 如 n=10
```

### 局部变量和全局变量
局部变量是指在当前代码块内可见的变量，使用`local`声明。例如下面的代码，将依次输出：111, 222, 111.
```bash
#! /bin/sh
num=111 # 全局变量
func1()
{
  local num=222 # 局部变量
  echo $num
}

echo "before---$num"
func1
echo "after---$num"
```

### 变量之间的运算

使用`expr`可以进行变量之间的运算，如下所示：

``` bash
# 注意要在操作符两边空余空格
expr 1 + 3
# 由于*是特殊字符，所以乘法要使用转义
expr 10 \* 2
```

### \`\`和""

使用\`\`（也就是TAB键上面的那个）包起来的部分，是可执行的命令。而使用""（引号）包起来的部分，是字符串。

``` bash
a=`expr 10 \* 3`
# output: 3
echo $a
# output: a
echo a
# output: expr 10 \* 3
a="expr 10 \* 3"
echo $a
```

另外，使用""（双引号）括起来的字符串会发生变量替换，而用''（单引号）括起来的字符串则不会。

``` bash
a=1
echo "$a"  # 输出 1
echo '$a'  # 输出 $a
```

### 读取输入

使用`read var1, var2, ...`的方式从键盘的输入读取变量的值。

``` bash
# input a=1
read a
# ouptut: 2
echo `expr $a + 1`
```

## 基本概念

### 命令的返回值

当bash命令成功执行后，返回给系统的返回值为`0`；否则为非零。可以据此判断上步操作的状态。使用`$?`可以取出上一步执行的返回值。

``` bash
# 将echo 错输为ecoh
ecoh "hello"
# output: 非零(127)
echo $?
# output: 0
echo $?
```

### 通配符

通配符是指`*`,`?`和`[...]`这三类。

`*`可以匹配任意多的字符，`?`用来匹配一个字符。`[...]`用来匹配括号内的字符。见下表。
![通配符](/img/shell-programming-wild-cards.jpg)

`[...]`表示法还有如下变形：

- 使用`-`用来指示范围。如`[a-z]`，表示`a`到`z`间任意一个字符。
- 使用`^`或`!`表示取反。如`[!a-p]`表示除了`a`到`p`间字符的其他字符。

### 输入输出重定向

重定向是指改变命令的输出位置。使用`>`进行输出重定向。使用`<`进行输入重定向。例如，`ls -l > a.txt`，将本目录下的文件信息输出到文本文件`a.txt`中，而不再输出到终端。

此外，`>>`同样是输出重定向。但是它会在文件末尾追加写入，不会覆盖文件的原有内容。

搭配使用`<`和`>`可以做文件处理。例如，`tr group1 group2`命令可以将`group1`中的字符变换为`group2`中对应位置的字符。使用如下命令：

``` bash
tr "[a-z]" "A-Z" < ori.txt > out.txt
```

可以将`ori.txt`中的小写字母转换为大写字母输出到`out.txt`中。

### 管道（pipeline）

管道`|`可以将第一个程序的输出作为第二个程序的输入。例如：

``` bash
cat ori.txt | tr "[a-z]" "A-Z"
```

会将`ori.txt`中的小写字母转换为大写，并在终端输出。

### 过滤器（Filter）

Filter是指那些输入和输出都是控制台的命令。通过Filter和输入输出重定向，可以很方便地对文件内容进行整理。例如：

``` bash
sort < names.txt | uniq > u_names.txt
```

`uniq`命令可以实现去重，但是需要首先对输入数据进行排序。上面的Filter可以将输入文件`names.txt`中的行文本去重后输出到`u_names.txt`中去。

## 控制流

### if 条件控制

在bash中使用`if`条件控制的语法和MATLAB等很像，要在末尾加上类似`end`的指示符，如下：

``` bash
if condition
then 
XXX
fi
```

或者加上`else`，使用如下的形式：
``` bash
if condition
then
    do something
elif condition
then
    do something
else
    do something
fi
```

那么，如何做逻辑运算呢？需要借助`test`关键字。

对于整数来说，我们可以使用`if test op1 oprator op2`的方式，判断操作数`op1`和`op2`的大小关系。其中，`operator`可以是`-gt`，`-eq`等。

或者另一种写法：`if [ op1 operator op2 ]`，但是注意后者`[]`与操作数之间有空格。
如下表所示（点击可放大）：

![比较整数的逻辑运算](/img/shell-programming-if-operators.jpg)

对于字符串，支持的逻辑判断如下：
![比较字符串的逻辑运算](/img/bash-programming-comparing-string.jpg)

举个例子，我们想判断输入的值是否为1或2，可以使用如下的脚本。注意`[]`的两边一定要加空格。
``` bash
#! /bin/bash
a=1
if [ $1=$a ]
then
    echo "you input 1"
elif [ $1=2 ]
then
    echo "you input 2"
else
    echo "you input $1"
fi
```


