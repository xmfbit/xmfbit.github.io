---
title: Python中的迭代器和生成器
date: 2017-04-21 15:53:23
tags:
     - python
---
在STL中，迭代器可以剥离算法和具体数据类型之间的耦合，使得库的维护者只需要为特定的迭代器（如前向迭代器，反向迭代器和随机迭代器）等实现算法即可，而不用关心具体的数据结构。在Python中，迭代器更是无处不在。这篇博客简要介绍Python中的迭代器和生成器，它们背后的原理以及如何实现一个自定义的迭代器/生成器，主要参考了教程[Iterators & Generators](http://anandology.com/python-practice-book/iterators.html)。

<!-- more -->

## 迭代器
使用`for`循环时，常常遇到迭代器。如下所示，可能是最常用的一种方式。

```
for i in range(100):
    # do something 100 times
```

在Python中，凡是可迭代的对象（Iterable Object），都可以用上面的方式进行迭代循环。例如，当被迭代对象是字符串时，每次得到的是字符串中的单个字符；当被迭代对象是文本文件时，每次得到的是文件中每一行的字符串；当被迭代对象是字典时，每次得到的是字典的`key`。

同样，也有很多函数接受的参数为可迭代对象。例如`list()`和`tuple()`，当传入的参数为刻碟带对象时，返回的是由迭代返回值组成的列表或者元组。例如

```
list({'x':1, 'y':2})  # => ['x', 'y']
```

为什么`list`或者`str`这样的可迭代对象能够被迭代呢？或者，自定义的类满足什么条件，就可以用`for x in XXX`这种方法来遍历了呢？

在Python中，有内建的函数`iter()`和`next()`。一般用法时，`iter()`方法接受一个可迭代对象，会调用这个对象的`__iter__()`方法，返回作用在这个可迭代对象的迭代器。而作为一个迭代器，必须有“迭代器的自我修养”，也就是实现`next()`方法（Python3中改为了`__next__()`方法）。

如下面的例子，`yrange_iter`是`yrange`的一个迭代器。`yrange`实现了`__iter__()`方法，是一个可迭代对象。调用`iter(yrange object)`的结果就是返回一个`yrange_iter`的对象实例。

```
# Version 1.0 使用迭代器类
class yrange_iter(object):
    def __init__(self, yrange):
        self.n = yrange.n
        self.i = 0
    def next(self):
        v = self.i
        self.i += 1
        return v

class yrange(object):
    def __init__(self, n):
        self.n = n
    def __iter__(self):
        return yrange_iter(self)

print type(iter(yrange(5))) # <class '__main__.yrange_iter'>
```

而不停地调用迭代器的`next()`方法，就能够不断输出迭代序列。如下所示：

```
In [3]: yiter = iter(yrange(5))

In [4]: yiter.next()
Out[4]: 0

In [5]: yiter.next()
Out[5]: 1

In [6]: yiter.next()
Out[6]: 2
```

其实，上面的代码略显复杂。在代码量很小，不是很在意代码可复用性时，我们完全可以去掉`yrange_iter`，直接让`yrange.__iter__()`方法返回其自身实例。这样，我们只需要在`yrange`类中实现`__iter__()`方法和`next()`方法即可。如下所示：

```
# Version2.0 简化版，迭代器是本身
class yrange(object):
    def __init__(self, n):
        self.n = n
    def __iter__(self):
        self.i = 0
        return self
    def next(self):
        v = self.i
        self.i += 1
        return v

In [8]: yiter = iter(yrange(5))

In [9]: yiter.next()
Out[9]: 0

In [10]: yiter.next()
Out[10]: 1

In [11]: yiter.next()
Out[11]: 2
```

然而，上述的代码仍然存在问题，我们无法指定迭代器生成序列的长度，也就是`self.n`实际上并没有用到。如果我只想产生0到10以内的序列呢？

我们只需要加入判断条件，当超出序列边界时，抛出Python内建的`StopIteration`异常即可。

```
# Version3.0 加入边界判断，生成有限长度序列
class yrange(object):
    def __init__(self, n):
        self.n = n
    def __iter__(self):
        self.i = 0
        return self
    def next(self):
        if self.i == self.n:
            raise StopIteration
        v = self.i
        self.i += 1
        return v

for i in yrange(5):
    print i
```

### Problem 1
Write an iterator class `reverse_iter`, that takes a `list` and iterates it from the reverse direction.

```
class reverse_iter(object):
    def __init__(self, alist):
        self.container = alist
        self.i = len(alist)

    def next(self):
        if self.i == 0:
            raise StopIteration
        self.i -= 1
        return self.container[self.i]
it = reverse_iter([1, 2, 3, 4])
```

## 生成器
生成器是一种方法，他指定了如何生成序列中的元素，生成器内部包含特殊的`yield`语句。此外，生成器函数是懒惰求值，只有当调用`next()`方法时，生成器才开始顺序执行，直到遇到`yield`语句。`yield`语句就像`return`，但是并未退出，而是打上断电，等待下一次`next()`方法的调用，再从上一次的断点处开始执行。我直接贴出教程中的代码示例。

```
>>> def foo():
        print "begin"
        for i in range(3):
            print "before yield", i
            yield i
            print "after yield", i
        print "end"

>>> f = foo()
>>> f.next()
begin
before yield 0
0
>>> f.next()
after yield 0
before yield 1
1
>>> f.next()
after yield 1
before yield 2
2
>>> f.next()
after yield 2
end
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
StopIteration
>>>
```

### 生成器表达式
生成器表达式和列表相似，将`[]`换为`()`即可。如下所示：

```
for i in (x**2 for x in [1,2,3,4]):
    print i
# print 1 4 9 16
```

生成器的好处在于惰性求值，这样一来，我们还可以生成无限长的序列。因为生成器本来就是说明了序列的生成方式，而并没有真的生成那个序列。

下面的代码使用生成器得到前10组勾股数。通过在调用`take()`方法时修改传入实参`n`的大小，该代码可以很方便地转换为求取任意多得勾股数。生成器的重要作用体现在斜边`x`的取值为$[0, \infty]$。如果不使用生成器，恐怕就需要写出好几行的循环语句加上`break`配合才可以达到相同的效果。

```
def integer(start, end=None):
    """Generate integer sequence [start, end)
       If `end` is not given, then [start, \infty]
    """
    i = start
    while True:
        if end is not None and i == end:
            raise StopIteration
        yield i
        i += 1

def take(n, g):
    i = 0
    while True:
        if i < n:
            yield g.next()
            i += 1
        else:
            raise StopIteration

# 假定 x>y>z，以消除两直角边互换的情况，如10, 6, 8和10, 8, 6
tup = ((x,y,z) for x in integer(0) for y in integer(0, x) for z in integer(0, y) if x*x==y*y+z*z)
list(take(10, tup))
```

### Problem 2
Write a program that takes one or more filenames as arguments and prints all the lines which are longer than 40 characters.

```
def readfiles(filenames):
    for f in filenames:
        for line in open(f):
            yield line

def grep(lines):
    return (line for line in lines if len(line)>40)

def printlines(lines):
    for line in lines:
        print line,

def main(filenames):
    lines = readfiles(filenames)
    lines = grep(lines)
    printlines(lines)
```

### Problem 3
Write a function `findfiles` that recursively descends the directory tree for the specified directory and generates paths of all the files in the tree.

注意`get_all_file()`方法中递归中生成器的写法，见SO的[这个帖子](http://stackoverflow.com/questions/248830/python-using-a-recursive-algorithm-as-a-generator
)。

```
import os

def generate_all_file(root):
    for item in os.listdir(root):
        item = os.path.join(root, item)
        if os.path.isfile(item):
            yield os.path.abspath(item)
        else:
            for item in generate_all_file(item):
                yield item

def findfiles(root):
    for item in generate_all_file(root):
        print item
```

### Problem 4
Write a function to compute the number of python files (.py extension) in a specified directory recursively.

```
def generate_all_py_file(root):
    return (file for file in generate_all_file(root) if os.path.splitext(file)[-1] == '.py')

print len(list(generate_all_py_file('./')))
```

### Problem 5
Write a function to compute the total number of lines of code in all python files in the specified directory recursively.

```
def generate_all_line(root):
    return (line for f in generate_all_py_file(root) for line in open(f))
print len(list(generate_all_line('./')))
```

### Problem 6
Write a function to compute the total number of lines of code, ignoring empty and comment lines, in all python files in the specified directory recursively.

```
def generate_all_no_empty_and_comment_line(root):
    return (line for line in generate_all_line(root) if not (line=='' or line.startswith('#')))

print len(list(generate_all_no_empty_and_comment_line('./')))
```

### Problem 7
Write a program `split.py`, that takes an integer `n` and a `filename` as command line arguments and splits the `file` into multiple small files with each having `n` lines.

```
def get_numbered_line(filename):
    i = 0
    for line in open(filename):
        yield i, line
        i += 1

def split(file_name, n):
    i = 0
    f = open('output-%d.txt' %i, 'w')
    for idx, line in get_numbered_line(file_name):
        f.write(line)
        if (idx+1) % n == 0:
            f.close()
            i += 1
            f = open('output-%d.txt' %i, 'w')

    f.close()
```

### Problem 9
The built-in function `enumerate` takes an `iteratable` and returns an `iterator` over pairs ``(index, value)`` for each value in the source.

Write a function `my_enumerate` that works like `enumerate`.

```
def my_enumerate(iterable):
    i = 0
    seq = iter(iterable)
    while True:
        val = seq.next()
        yield i, val
        i += 1
```
