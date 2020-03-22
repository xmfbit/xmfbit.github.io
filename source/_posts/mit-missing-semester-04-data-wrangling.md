---
title: MIT Missing Semester - Data Wrangling
date: 2020-03-15 21:47:18
tags:
    - tool
---

这是[MIT Missing Semester系列](https://missing.csail.mit.edu/2020/data-wrangling/)的第四讲。关于vim的第三讲跳过。Data Wrangling在这里的意思是对数据做变换（Transformation）。例如将一个MP4格式的视频转换为AVI，或或者是从日志中提取所需要的结构化文本信息。具体到本课，主要是处理文本信息：如何匹配到我们感兴趣的信息，如果构建一个处理的pipeline等。

<!-- more-->
`
## 正则表达式

在很久以前，总结了一篇关于python中的正则表达式的常用用法，竟然也是博客的第一篇文章：[python正则表达式](https://xmfbit.github.io/2014/07/17/python-reg-exp/)。

推荐一个[交互式的正则表达式学习网站](https://regexone.com/)。这里有一些简单的规则：

- `.`匹配任意字符，除了`\n`
- `*`匹配前缀的任意个，包括0个
- `+`匹配前缀的任意个，不包括0个
- `?`匹配前缀的0个或1个
- `[abc]`匹配给定集合里面的元素，例如这里匹配`a`或`b`或`c`
- `(ab)`匹配给定的组合，例如这里匹配`ab`
- `(exp1|exp2)`匹配`exp1`或`exp2`
- `^`指示一行的开头
- `$`指示一行的结尾

要注意的是，`(`在下面的sed中如果没有特殊说明，被视为普通字符，需要加上`-E`选项。

如果我们想要指定具体的次数呢？可以使用`.{n}`的形式。例如，`a{3}`表明匹配3个`a`；`[ab]{4}`匹配4个`a`或`b`。使用range表达式，表明在某个范围内：`.{2,5}`表示2到5个任意字符。

## sed

sed(Stream Editor)可以帮助我们变换文本。sed每次从输入流中读入一行，作相应变换，并输出。

> A stream editor is used to perform basic text transformations on an input stream (a file or input from a pipeline)

![sed workflow](/img/sed_workflow.png)

[这里](https://www.tutorialspoint.com/sed/index.htm)是一个sed的教程，下面结合该教程和讲师的实例，对sed使用做一说明。

### sed的其他用法

这里首先对sed的其他用法做一说明。

使用`-e`可以传入一些命令，例如`1d`就是删除第一行。通过串联，可以删除多行。

``` bash
# 删除第一行和第五行
sed -e '1d' -e `5d` input
```

还可以使用`-f`指示从某个文件内读取命令，

``` bash
echo "1d\n2d" > arg.txt
sed -f arg.txt input
```

### sed对文本进行查找替换

sed最常用的场景之一，使用如下命令，将文本文件中的`pattern`（一个正则表达式）替换为`new`。当`new`为空时，将直接删去`pattern`。最后的`g`如果不加，则只匹配一次，加上`g`表示全局。

``` bash
sed 's/pattern/new/ filename/g'
```

例如：

``` bash
# .* 表示任意多个任意字符，包括0个。所以下面会把 says hello以及它前面的内容都删掉
echo "cat says hello to dog" | sed 's/.*says hello//'
# output: to dog
```

注意，`.*`组合是greedy的。这意味着它会尽可能多地去匹配任意字符。

``` bash
echo "cat says hello to says hello to dog" | sed 's/.*says hello//'
#output: to dog
```

### capture group

capture group是指我希望记住匹配到的值。在正则表达式中，使用`()`括起来的就是capture group。我们可以使用`\1`，`\2`来引用它们。

``` bash
# 想知道cat对谁打招呼了
# 我们使用`(.*)`来匹配任意多的字符，也就是dog
# 并将整行替换为`\1`，也就是capture group
echo "cat says hello to dog" | sed -E 's/.*says hello to (.*)/\1/'
# output：dog
```

## sort和uniq

sort，顾名思义，读入input，排序，再将它们输出。uniq，可以将**紧邻的**相同行进行合并。因为uniq只能合并紧邻的向同行，所以常常和sort配合使用。

``` bash
# -c 会在每一行前面加上一列，显示重复出现的次数
# 继续排序，并输出出现频率最高的10个 （sort升序排列，使用tail找到末尾，也就是最大的）
# sort -k        -k, --key=KEYDEF
#                sort via a key; KEYDEF gives location and type
#      -n        numeric sort
xxx | sort | uniq -c | sort -nk1,1 | tail -10
```

## awk

awk是另一种stream editor。与sed相比，它更针对于成列的数据。例如，我们可以打印文本文件的第一列：

``` bash
awk '{print $1}' input
```

awk的一般用法还会加上`pattern`，例如`awk 'pattern {action}' input`。例如：

``` bash
# $0表示非特定列，而是整行
# 找出第一列符合给定pattern的那些行，并打印这个整行
awk '$1 ~ /pattern/ {print $0}' input

echo "hello world\nsmile world" | awk '$1 ~ /^h.*o$/ {print $1}'
# output: hello
# 这里smile这行因为不符合pattern，所以被filter掉了
```

awk的功能还远不止此。awk中可以使用循环，分支等语句构成更复杂的逻辑。

paste命令可以用来合并多行为一行。下面的命令将输入文件的第一列顺序拼接为一行，并使用逗号分隔。

``` bash
awk '{print $1}' input | paste -sd,
```

> The paste utility concatenates the corresponding lines of the given input files, replacing all but the last file's newline characters with a single tab character, and writes the resulting lines to standard output.

## 总结

本课主要介绍了一些常用的文本处理命令，包括sed, awk, sort, uniq, paste等。下面使用tldr命令给出这些命令的常用用法供参考。

### sed

``` bash
➜  ~ tldr sed
# sed

  Edit text in a scriptable manner.

- Replace the first occurrence of a regular expression in each line of a file, and print the result:

  sed 's/regex/replace/' filename

- Replace all occurrences of an extended regular expression in a file, and print the result:

  sed -r 's/regex/replace/g' filename

- Replace all occurrences of a string in a file, overwriting the file (i.e. in-place):

  sed -i 's/find/replace/g' filename

- Replace only on lines matching the line pattern:

  sed '/line_pattern/s/find/replace/' filename

- Delete lines matching the line pattern:

  sed '/line_pattern/d' filename

- Print only text between n-th line till the next empty line:

  sed -n 'n,/^$/p' filename

- Apply multiple find-replace expressions to a file:

  sed -e 's/find/replace/' -e 's/find/replace/' filename

- Replace separator / by any other character not used in the find or replace patterns, e.g., #:

  sed 's#find#replace#' filename

- Print only the n-th line of a file:

  sed 'nq;d' filename
```

### awk

``` bash
➜  ~ tldr awk
# awk

  A versatile programming language for working on files.

- Print the fifth column (a.k.a. field) in a space-separated file:

  awk '{print $5}' filename

- Print the second column of the lines containing "something" in a space-separated file:

  awk '/something/ {print $2}' filename

- Print the last column of each line in a file, using a comma (instead of space) as a field separator:

  awk -F ',' '{print $NF}' filename

- Sum the values in the first column of a file and print the total:

  awk '{s+=$1} END {print s}' filename

- Sum the values in the first column and pretty-print the values and then the total:

  awk '{s+=$1; print $1} END {print "--------"; print s}' filename

- Print every third line starting from the first line:

  awk 'NR%3==1' filename
```

### sort

``` bash
➜  ~ tldr tldr
# sort

  Sort lines of text files.

- Sort a file in ascending order:

  sort filename

- Sort a file in descending order:

  sort -r filename

- Sort a file in case-insensitive way:

  sort --ignore-case filename

- Sort a file using numeric rather than alphabetic order:

  sort -n filename

- Sort the passwd file by the 3rd field, numerically:

  sort -t: -k 3n /etc/passwd

- Sort a file preserving only unique lines:

  sort -u filename

- Sort human-readable numbers (in this case the 5th field of `ls -lh`):

  ls -lh | sort -h -k 5
```

### uniq

``` bash
➜  ~ tldr uniq
# uniq

  Output the unique lines from the given input or file.
  Since it does not detect repeated lines unless they are adjacent, we need to sort them first.

- Display each line once:

  sort file | uniq

- Display only unique lines:

  sort file | uniq -u

- Display only duplicate lines:

  sort file | uniq -d

- Display number of occurrences of each line along with that line:

  sort file | uniq -c

- Display number of occurrences of each line, sorted by the most frequent:

  sort file | uniq -c | sort -nr
```

### paste

```
➜  ~ tldr paste
# paste

  Merge lines of files.

- Join all the lines into a single line, using TAB as delimiter:

  paste -s file

- Join all the lines into a single line, using the specified delimiter:

  paste -s -d delimiter file

- Merge two files side by side, each in its column, using TAB as delimiter:

  paste file1 file2

- Merge two files side by side, each in its column, using the specified delimiter:

  paste -d delimiter file1 file2

- Merge two files, with lines added alternatively:

  paste -d '\n' file1 file2
```

## Exercise

### 关于正则匹配的应用

- Find the number of words (in `/usr/share/dict/words`) that contain at least three `a`s and don’t have a `'s` ending.

``` bash
cat /usr/share/dict/words | grep -E '(.*a){3}' | grep -Ev "\'s$" | wc -l
#    7047
```

### sed 使用

就地修改文件，使用`-i`。最好在后面指定backup文件的后缀名，否则将不会做backup。有丢失原始数据的风险。

``` bash
-i extension
         Edit files in-place, saving backups with the specified extension.  If a zero-length extension is given, no 
         backup will be saved. It is not recommended to give a zero-length extension when in-place editing files, as 
         you risk corruption or partial content in situations where disk space is exhausted, etc.
```
