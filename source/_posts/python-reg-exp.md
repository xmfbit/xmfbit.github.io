---
title: Python Regular Expressions （Python 正则表达式)
date: 2014-07-17 19:00:00
tags: 
    - python
---

本文来自于Google Developers中对于Python的介绍。[https://developers.google.com/edu/python/regular-expressions](https://developers.google.com/edu/python/regular-expressions "Google Python Class, Regular Expression")。



## 认识正则表达式 ##

Python的正则表达式是使用 **re 模块**的。


``` py    
    match = re.search(pattern,str)
    if match:
    	print 'found',match.group()
    else:
        print 'NOT Found!'
        
```

## 正则表达式的规则 ##

### 基本规则 ###
- a, x, 9 都是普通字符 (ordinary characters)
- . (一个点)可以匹配任何单个字符（除了'\n'）
- \w（小写的w）可以匹配一个单词里面的字母，数字或下划线 [a-zA-Z0-9_];\W （大写的W）可以匹配非单词里的这些元素
- \b 匹配单词与非单词的分界
- \s（小写的s）匹配一个 whitespace character，包括 space，newline，return，tab，form(\n\r\t\f)；\S（大写的S）匹配一个非 whitespace character
- \d 匹配十进制数字 [0-9]
- ^=start，$=end 用来匹配字符串的开始和结束
- \ 是转义字符，用 \. 来匹配串里的'.'，等
### 一些基本的例子 ###

``` py
    ## 在字符串'piiig'中查找'iii'
    match = re.search(r'iii', 'piiig')  # found, match.group() == "iii"
    match = re.search(r'igs', 'piiig')  #  not found, match == None

    ## . 匹配除了\n的任意字符
    match = re.search(r'..g', 'piiig')  #  found, match.group() == "iig"

    ## \d 匹配0-9的数字字符, \w 匹配单词里的字符
    match = re.search(r'\d\d\d', 'p123g') #  found, match.group() == "123"
    match = re.search(r'\w\w\w', '@@abcd!!') #  found, match.group() == "abc"   
```

### 重复 ###
可以用'+' '*' '?'来匹配0个，1个或多个重复字符。

- '+' 用来匹配1个或者多个字符
- '*' 用来匹配0个或者多个字符
- '?' 用来匹配0个或1个字符

注意，'+'和'*'会匹配尽可能多的字符。

### 一些重复字符的例子 ###

``` py
    ## i+  匹配1个或者多个'i'
    match = re.search(r'pi+', 'piiig') #  found, match.group() == "piii"

    ## 找到字符串中最左边尽可能长的模式。
    ## 注意，并没有匹配到第二个 'i+'
    match = re.search(r'i+', 'piigiiii')  #  found, match.group() == "ii"

    ## \s*  匹配0个或1个空白字符 whitespace
    match = re.search(r'\d\s*\d\s*\d', 'xx1 2   3xx')  #  found, match.group() == "1 2   3"
    match = re.search(r'\d\s*\d\s*\d', 'xx12  3xx')    #  found, match.group() == "12  3"
    match = re.search(r'\d\s*\d\s*\d', 'xx123xx')      # found, match.group() == "123"

    ## ^ 匹配字符串的第一个字符
    match = re.search(r'^b\w+', 'foobar')  # not found, match == None
    ## 与上例对比
    match = re.search(r'b\w+', 'foobar')   # found, match.group() == "bar"
```

### Email ###
考虑一个典型的Email地址：someone@host.com，可以用如下的方式匹配：

``` py
    match = re.search(r'\w+@\w+',str)
```    

但是，对于这种Email地址 'xyz alice-b@google.com purple monkey' 则不能奏效。

### 使用方括号 ###
方括号里面的字符表示一个字符集合。[abc]可以被用来匹配'a'或者'b'或者'c'。\w \s等都可以用在方括号里，除了'.'以外，它只能用来表示字面意义上的‘点’。所以上面的Email规则可以扩充如下：
    
``` py
    match = re.search('r[\w.-]+@[\w.-]+',str)
```

你还可以使用'-'来指定范围，如[a-z]指示的是所有小写字母的集合。所以如果你想构造的字符集合中有'-'，请把它放到末尾[ab-]。另外，前方加上'^'，用来表示取集合的补集，例如[^ab]表示除了'a'和'b'之外的其他字符。

## 操作 ##
以Email地址为例，如果我们想要分别提取该地址的用户名'someone'和主机名'host.com'该怎么办呢？
可以在模式中用圆括号指定。

``` py
    str = 'purple alice-b@google.com monkey dishwasher'
    match = re.search('([\w.-]+)@([\w.-]+)', str)   #用圆括号指定分割
    if match:
        print match.group()   ## 'alice-b@google.com' (the whole match)
        print match.group(1)  ## 'alice-b' (the username, group 1)
      	print match.group(2)  ## 'google.com' (the host, group 2)
```

### findall 函数
与group函数只找到最左端的一个匹配不同，findall函数找到字符串中所有与模式匹配的串。

``` py
    str = 'purple alice@google.com, blah monkey bob@abc.com blah dishwasher'
    ## findall返回一个包含所有匹配结果的 list
    emails = re.findall(r'[\w\.-]+@[\w\.-]+', str) ## ['alice@google.com', 'bob@abc.com']
    for email in emails:
        print email
```

### 在文件中使用findall
当然可以读入文件的每一行，然后对每一行的内容调用findall，但是为什么不让这一切自动发生呢？

``` py
	f = open(filename.txt,'r')
	matches = re.findall(pattern,f.read())
```

### findall 和分组
和group的用法相似，也可以指定分组。

``` py
    str = 'purple alice@google.com, blah monkey bob@abc.com blah dishwasher'
    ##　返回了一个list
    tuples = re.findall(r'([\w\.-]+)@([\w\.-]+)', str)
    print tuples  ## [('alice', 'google.com'), ('bob', 'abc.com')]
    ##　list中的元素是tuple 
    for tuple in tuples:
      print tuple[0]  ## username
      print tuple[1]  ## host
```

## 调试 ##

正则表达式异常强大，使用简单的几条规则就可以演变出很多的模式组合。在确定你的模式之前，可能需要很多的调试工作。在一个小的测试集合上测试正则表达式。

## 其他选项

正则表达式还可以设置“选项”。

``` py
    match = re.search(pat,str,opt)
```

这些可选项如下：

- IGNORECASE  忽视大小写
- DOTALL  允许'.'匹配'\n'
- MULTILINE  在一个由许多行组成的字符串中，允许'^'和'$'匹配每一行的开始和结束
