---
title: Effective CPP 阅读 - Chapter 1 让自己习惯C++
date: 2017-04-20 13:49:42
tags:
     - cpp
---
本系列是《Effective C++》一书的阅读摘记，分章整理各个条款。
![写C++需要人品](/img/effectivecpp_01_cpp_rely_on_renpin.jpg)
<!-- more-->

### 01 将C++视作语言联邦
C++在发明之初，只是一个带类的C。现在的C++已经变成了同时支持面向过程，面向对象，支持泛型和模板元编程的巨兽。这部分内容可以参见“C++的设计与演化”一书。

本条款中，将C++概括为四个次语言组成的联邦：
- 传统C：面向过程，也规定了C++基本的语法。
- OOP：面向对象，带类的C，加入了继承，虚函数等概念。
- Template：很多针对模板需要特殊注意的条款，甚至催生了模板元编程。
- STL：标准模板库。使用STL要遵守它的约定。

想要高效地使用C++，必须根据不同的情况遵守不同的编程规范。

### 02 尽量使用`const`, `enum`, `inline`替换 `#define`（以编译器替换预处理器）

`#define`是C时代遗留下来的预编译指令。

当`#define`用来定义某个常量时，通常`const`是一个更好的选择。

``` cpp
#define PI 3.14
const double PI = 3.14;
```
当此常量为整数类型（`int`, `char`, `bool`）等时，也可以使用`enum`定义常量。这种做法常常用在模板元编程中。

``` cpp
enum {K = 1};
```
对于`const`常量，你可以获取变量的地址，但是对于`enum`来说，无法获取变量的地址。对于这一点来说，`enum`和`#define`相类似。

另一种可能使用`#define`的场景是宏定义。这种情形可以使用`inline`声明内联函数解决。

总之，尽可能相信编译器的力量。使用`#define`将遮蔽编译器的视野，带来奇怪的问题。

### 03 尽可能使用`const`
`const`不止是给程序员看的，而且为编译器指定了一个语义约束，即这个对象是不该被改变的。所以任何试图修改这个对象的操作，都会被编译器检查出来，并给出error。

所以，如果某一变量满足`const`的要求，那么请加上`const`，和编译器签订一份契约，保护你的行为。

这里不再讨论`const`的寻常用法。提示一下：当修饰指针变量时，`const`在星号左边，是指指针所指物是常量；当`const`在星号右边，是指指针本身是常量。如下所示：

``` cpp
const int* p = &a;
*p = 5;   // 非法
p = &b;   // 合法
int* const p = &a;
p = &b;    // 非法
*p = 5;    // 合法
```

STL中，如果声明某个迭代器为`const`，是指该迭代器本身是常量；如果你的意思是迭代器指向的元素为常量，那么使用`const_iterator`。

`const`更丰富的用法是用于函数声明中，

- 当修饰返回值时，意思是返回值不能修改。这可以让你避免无意义的赋值，尤其是以下的错误：

``` cpp
if (fun(a, b) = c)  // 这里错把 == 打成了 =
```

- 当修饰参数时，常常用做 pass-by-const-reference 的形式，不再多说了。
- 当修饰函数本身时，常常用在类中的成员函数上，意思是这个函数将不改变对象的成员。

这种情况下，可能会有`const`重载现象。

``` cpp
class my_string{
  const char& operator[](size_t pos) {
	  return this->ptr[pos];
  }
  char& operator[](size_t pos) {
	  return this->ptr[pos];
  }
};
```

实际调用时，根据调用该函数的对象是否是`const`的来决定究竟调用哪个版本。

上面的实现未免过于复杂，我们还可以改成下面的形式：

``` cpp
class my_string{
  const char& operator[](size_t pos) {
	  return this->ptr[pos];
  }
  char& operator[](size_t pos) {
	  return const_cast<char&>(
	         static_cast<const my_string&>(*this)[pos]);
  }
};
```

注意上面的代码进行了两次类型转换。由`non-const reference`转为`const reference`是类型安全的，使用`static_cast`进行。最后我们要脱掉`const char&`的`const`属性，使用了`const_cast`。

对于`const`成员函数，有时不得不修改类中的某些成员变量，可以将这些变量声明为`mutable`。

### 04 确保对象在使用前已经被初始化

使用未被初始化的变量有可能导致未定义的行为，导致奇怪的bug。所以推荐为所有变量进行初始化。

对于内建类型，需要手动初始化。

对于用户自定义类型，一般需要调用构造函数初始化。推荐在构造函数中使用初始化列表进行初始化，这样可以避免不必要的性能损失。原因见下：

``` cpp
public A(name, age) {
  this->name = name; // 这是赋值，不是初始化！
  this->age = age;
}
```

如果在类`A`的构造函数中使用初始化列表，就可以避免上面的赋值，而是使用`copy-construct`实现。

需要注意，成员初始化的顺序与其在类中声明的顺序相同，与初始化列表中的顺序无关。所以推荐将两者统一。

讨论完上述情况，再来看一种特殊变量：不同编译单元`non-local static`变量，是指不在某个函数scope下的`static`变量。这种变量的初始化顺序是未定义的，所以作者推荐使用单例模式，将它们移动到某个函数中去，明确初始化顺序。这里不再多说了。
