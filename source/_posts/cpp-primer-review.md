---
title: 重读 C++ Primer
date: 2019-05-01 13:32:58
tags:
   - cpp
---
重读C++ Primer第五版，整理一些糊涂的语法知识点。

<div  align="center">    
 <img src="/img/cpp_is_terrible.jpg" width = "200" height = "300" alt="入门到放弃" align=center />
</div>

<!-- more -->
## 基础语法

总结一些比较容易搞乱的基础语法。

### const 限定说明符

- `const`对象一般只在当前文件可见，如果希望在其他文件访问，在声明和定义时，均需加上`extern`关键字。

``` cpp
extern const int BUF_SIZE = 100;
```

- 顶层`const`和底层`const`

指针本身也是对象，所以有所谓的“常量指针”（指针本身不能赋值）和“指向常量的指针”（指针指向的那个对象不能赋值）。

``` cpp
int a = 10;
// 指针指向的对象不能经由指针赋值
const int* p1 = &a;
*p = 0;  // 错误
// 指针本身不能再赋值
int* const p2 = &a;
int b = 0;
p2 = &p;   // 错误
```

如何记住这条规则？c++中类型说明从右向左读即可。例如`p1`，其左侧首先遇到`int*`，故其是个“普通”指针（没有被`const`修饰），再往左才读到`const`，故这个指针指向的内容是常量，不能修改。`p2`同理。

把“指针本身是常量”的行为称为“顶层const”（top-level），把“指针指向内容是常量”的行为称为“底层const”（low-level）。

### auto 和 decltype

- `auto`类型推断的规则

编译器推断`auto`声明变量的类型时，可能和初始值类型不一样。当初始值类型为引用时，编译器以被引用对象的类型作为`auto`的类型，除非显式指明。

``` cpp
int i = 0;
int& ri = i;
// type of j: int
auto j = ri;
// type of rj: int&
auto& rj = ri;
```

另外，`auto`只会保留底层`const`，忽略顶层`const`，除非显式指定。

``` cpp
int a = 0;
const int* const p = &a;
// type of b: int
auto b = a;
// type of p1: const int*
auto p1 = p;
// type of p2: const int* const
const auto p2 = p;
p1 = &b;   // ok, p1 本身已经不是const的了
p2 = &b;   // wrong! 显式指定了 p2 本身是 const
*p1 = 10;  // wrong! p1 保留了底层const，指向的内容仍然不可改变
```

- `decltype` 类型推断规则

和`auto`不同，`decltype`保留表达式的顶层`const`和引用。

1. 如果表达式是变量，那么返回该变量的类型；
2. 如果表达式不是纯变量，返回表达式结果的类型；
3. 如果表达式是解引用，返回引用类型。

``` cpp
int i = 42, *p = &i, &r = i;
decltype(i) j;    // ok, j is a int
decltype(r) y;    // wrong! y是引用类型，必须初始化
decltype(r + 0) z;  // ok, r+0 返回值是int
decltype(*p) c;   // wrong! 解引用的结果是引用，必须初始化
```

有一种情况特殊，如果是春变量，但是变量名加上括号，结果将是引用。原因：变量加上括号，将会被当做表达式。而变量又可以被赋值，所以得到了引用。

``` cpp
dectype((i)) d;  // wrong! d是引用
```

## 泛型算法

C++的标准库中实现了很多泛型算法，如`find`, `sort`等。它们大多定义在`<algorithm>`头文件中，一些数值相关的定义在`<numeric>`中。通过“迭代器”这一层抽象，泛型算法可以不关心所操作数据实际储存的容器，不过仍然受制于实际数据类型。例如`find`中，为了比较当前元素是否为所求值，要求元素类型实现`==`运算。好在这些算法大多支持自定义操作。

### 迭代器

在标准库的`<iterator>`中，定义了如下几种通用迭代器。

- 插入迭代器

插入器是一个迭代器的适配器，接受一个容器，生成一个用于该容器的迭代器，能够实现向该容器插入元素。插入迭代器有三种，区别在于插入元素的位置：

1. `back_inserter`，创建一个使用`push_back`插入的迭代器
2. `front_inserter`，创建一个使用`push_front`插入的迭代器
3. `inserter`，创建一个使用`insert`的迭代器，在给定的迭代器前面插入元素

``` cpp
#include <iostream>
#include <vector>
#include <iterator>
#include <algorithm>
using namespace std;

// 使用back_inserter插入数据
int main() {
    int a[] = {1,2,3,4,5};
    vector<int> b;
    // copy a -> b, 动态改变b的大小
    copy(begin(a), end(a), back_inserter(b));
    for (auto v: b) {
        cout << v << endl;
    }
    // b: 1, 2, 3, 4, 5
    return 0;
}

// 使用inserter，将数据插入指定位置
int main() {
    int a[] = {1,2,3,4,5};
    vector<int> b {6,7,8};
    // find iter of value 8
    auto iter = find(b.begin(), b.end(), 8);
    // copy a -> b before value 8
    copy(begin(a), end(a), inserter(b, iter));
    for (auto v : b) {
        cout << v << endl;
    }
    // b: 6, 7, 1, 2, 3, 4, 5, 8
    return 0;
}
```

这里要注意的是，当使用`front_inserter`时，由于插入总是在容器头部发生，所以最后的插入结果是原始数据序列的逆序。

- 流迭代器

虽然输入输出流不是容器，不过也有用于这些IO对象的迭代器：`istream_iterator`和`ostream_iterator`。这样，我们可以通过它们向对应的输入输出流读写数据，

创建输入流迭代器时，必须指定其要操作的数据类型，并将其绑定到某个流（标准输入输出流或文件流），或使用默认初始化，得到当做尾后值使用的迭代器。

``` cpp
istream_iterator<int> in_iter(cin);
istream_iterator<int> in_eof;
// 使用迭代器构建vector
vector<int> values(in_iter, in_eof);
```

创建输出流迭代器时，必须指定其要操作的数据类型，并向其绑定到某个流，还可以传入第二个参数，类型是C风格的字符串（字符串字面常量或指向`\0`结尾的字符数组指针），表示在输出数据之后，还会输出此字符串。

``` cpp
vector<int> v{1,2,3,4,5};
// 输出：1       2       3       4       5 
copy(v.begin(), v.end(), ostream_iterator<int>(cout, "\t"));
```

- 反向迭代器

顾名思义，反向迭代器的迭代顺序和正常的迭代器是相反的。使用`rbegin`和`rend`可以获得绑定在该容器的反向迭代器。不过`forward_list`和流对象，由于没有同时实现`++`和`--`，所以没有反向迭代器。

反向迭代器常常用来在容器中查找最后一个满足条件的元素。这时候要注意，如果继续使用该迭代器，顺序仍然是反向的。如果需要正向迭代器，可以使用`.base()`方法得到对应的正向迭代器。不过要注意，正向迭代器和反向迭代器的位置会不一样哦~

``` cpp
// 找到数组中最后一个5,并将其后数字打印出来
vector<int> v {10, 5, 4, 5, 1, 2};
auto iter = find(v.rbegin(), v.rend(), 5);
// 输出：5,4,5,10,
copy(iter, v.rend(), ostream_iterator<int>(cout, ","));
cout << "\n";
// 输出：1,2, 注意并没有输出5
copy(iter.base(), v.end(), ostream_iterator<int>(cout, ","));
```

## 未完待续

拖延症发作。。。