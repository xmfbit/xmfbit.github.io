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
