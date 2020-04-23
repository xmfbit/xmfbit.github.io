---
layout: post
title: MIT OCW - Effective Pragramming in C/C++ 0x00
date: 2020-04-22 00:41:32
tags:
    - 公开课
    - ocw
    - 6.S096
---

这是我第一次在MIT OCW上学习课程。第一门先从自己较有把握，又和实际工作联系较紧密的C/C++编程课开始。[这门课](https://ocw.mit.edu/courses/electrical-engineering-and-computer-science/6-s096-effective-programming-in-c-and-c-january-iap-2014/index.htm)的编号为6.S096，为EECS的本科生开设。当前版本为2014年。

<!-- more -->

# 课程概览

- 快节奏地介绍C/C++编程语言，强调良好的编程实践和效率
- 主要覆盖下面的topic：
  - OOP
  - 内存管理
  - C/C++的优势
  - 优化（效率）
- 学生应该有一些编程经验，也就是说这不是一门编程入门课，但可以没有C/C++的基础
- 这是一门IAP（看起来是4周的国内所谓的“小学期“），一般是持续整个1月

# 课程安排

![课程日历](/img/mit_ocw_effective_cpp_calendar.png)

# 课前自测

这里把几道课前自测的代码题解答写在这里：

## 数据结构quiz

- queue和stack的区别：前者FIFO，后者FILO
- heap 数据结构：[heap](https://en.wikipedia.org/wiki/Heap_(data_structure))是一个完全二叉树，且满足每个节点的父节点的值都不小于（或不大于）该节点的值。其中，前者称为大根堆，后者称为小根堆
- hash table / dict：[hash table](https://en.wikipedia.org/wiki/Hash_table)利用hash函数计算插入值的索引，建立key-value的映射关系。能够在$O(1)$时间复杂度下实现查找

## 判断素数

判断$[2, \sqrt{x}]$内是否有能够整除$x$的。

``` cpp
#include <iostream>
#include <cmath>
using namespace std;

bool is_prime(int x) {
    int sqrt_x = static_cast<int>(std::sqrt(x));
    for (int i = 2; i <= sqrt_x; ++i) {
        if (x % i == 0) return false;
    }
    return true;
}

int main() {
    cout << is_prime(2) << endl;
    cout << is_prime(4) << endl;
    cout << is_prime(7) << endl;
    cout << is_prime(9) << endl;
    cout << is_prime(100) << endl;
    return 0;
}
```

参考答案：

``` cpp
bool is_prime( int n ) {
  if( n <= 2 || n % 2 == 0 ) {
    return ( n == 2 );
  }
  for( int i = 3; i * i <= n; i += 2 ) {
    if( n % i == 0 ) {
      return false;
    }
  }
  return true;
}
```

## 升序打印全排列

这个代码没有做升序，需要在得到后$n-1$个数字的全排列之后，安排`push_back`的顺序，先不想了。。

``` cpp
#include <iostream>
#include <iterator>
#include <vector>
using namespace std;

// permutation of intergers s,s+1,s+2,...,e
vector<vector<int>> permutation_helper(int s, int e) {
    vector<vector<int>> ret;
    if (s > e) {
        return ret;
    }
    if(s == e) {
        ret.push_back({s});
        return ret;
    }
    auto part = permutation_helper(s+1, e);
    for (const auto& p: part) {
        for (int i = 0; i <= p.size(); ++i) {
            ret.push_back({});
            std::copy(p.begin(), p.begin()+i, back_inserter(ret.back()));
            ret.back().push_back(s);
            std::copy(p.begin()+i, p.end(), back_inserter(ret.back()));
        }
    }
    return ret;
}


void print_permutation(int n) {
    auto ret = permutation_helper(1, n);
    for (const auto& array: ret) {
        for (auto v: array) {
            cout << v << " ";
        }
        cout << endl;
    }
}

int main() {
    print_permutation(3);
    return 0;
}
```

参考答案：

``` cpp
void swap( int *a, int *b ) {
  int t = *a;
  *a = *b;
  *b = t;
}
 
void permute( int *digits, int n, int p ) {
  if( p == n ) {
    for( int i = 0; i < n; ++i ) {
      printf( "%d ", digits[i] );
    }
    printf( "\n" );
  } else {
    for( int i = p; i < n; ++i ) {
      swap( &digits[p], &digits[i] );
      permute( digits, n, p + 1 );
      swap( &digits[p], &digits[i] );
    }
  }
}
 
void print_permutations( int n ) {
  int *digits = malloc( n * sizeof( int ) );
  for( int i = 0; i < n; ++i ) {
    digits[i] = i + 1;
  }
  permute( digits, n, 0 );
  free( digits );
}
```

## 线性查找和二分查找

``` cpp
#include <iostream>
#include <vector>
using namespace std;

int linear_search(const vector<int>& arr, int val) {
    for (int i = 0; i < arr.size(); ++i) {
        if (arr[i] == val) {
            return i;
        }
    }
    return -1;
}

int binary_search(const vector<int>& arr, int val) {
    int lo = 0;
    int hi = arr.size();
    while (lo < hi) {
        int mid = lo + (hi - lo) / 2;
        if (arr[mid] < val) lo = mid + 1;
        else if (arr[mid] > val) hi = mid;
        else return mid;
    }
    return -1;
}

int main()
{
    cout << binary_search({1}, 1) << endl;
    cout << binary_search({1}, 0) << endl;
    cout << binary_search({1,2,3,4}, 4) << endl;
    cout << binary_search({1,2,3,4}, 2) << endl;
    return 0;
}
```

# Best Practice

## 整体上

- 保持良好而统一的代码风格
- 代码应该自成文档：代码是给人读的，注意代码的可读性，例如变量名，函数名等
- 保持头文件简洁干净，不要引入不必要的系统头文件
- 不要暴露私有成员变量和方法，避免友元
- 在能用该用`const`的地方用：不改变成员变量的成员函数是`const`的；接受引用而不改变其值的函数，应在参数列表使用`const`
- 写可移植代码，尽量不用编译器特性，例如`long`或`unsigned`，考虑使用`size_t`替代
- 不要泄露内存，例如每一个`new`出来的堆上内存都要用`delete`释放掉

## 类的设计

- 遵从[rule of three](https://en.wikipedia.org/wiki/Rule_of_three_%28C%2B%2B_programming%29)：如果你的类需要一个non-trival的析构函数，意味着你要么需要实现自己的拷贝构造（copy constructor）和拷贝赋值（copy assignment）函数，要么要disable掉它们。如果需要，实现移动构造（move constructor）和移动赋值（move assignment）函数
- 不要使用全局可见的数据，可以将其封装为类，并使用适当的接口和它交互
- 在构造函数中使用初始化列表（initializer list）

## 其他

- 使用`nullptr`，而不是旧风格的`NULL`
- 只要对代码清晰性有利，就使用`auto`：[AAA style, or triple-A style](https://herbsutter.com/2013/08/12/gotw-94-solution-aaa-style-almost-always-auto/)
- 统一的初始化语法，例如尽可能都使用初始化列表`{xxx}`
- 将`*`和`&`紧跟变量，而不是类型
- 善于使用STL

About AAA-style:

> Guideline: Remember that preferring auto variables is motivated primarily by correctness, performance, maintainability, and robustness—and only lastly about typing convenience.

如果我确实需要写明变量的类型呢？推荐：

> Guideline: Consider declaring local variables auto x = type{ expr }; when you do want to explicitly commit to a type. It is self-documenting to show that the code is explicitly requesting a conversion, it guarantees the variable will be initialized, and it won’t allow an accidental implicit narrowing conversion. Only when you do want explicit narrowing, use ( ) instead of { }.

使用`auto`无法方便地得知变量的类型，使得代码可读性下降？

- 使用IDE可以避免这个问题
- 关注接口，而非实现；一个有点令人惊讶的发现，在写C++时，我们并没有想象中的那样关心变量的类型。

> Q: What does “write code against interfaces, not implementations” mean, and why is it generally beneficial?

> A: It means we should care principally about “what,” not “how.” This separation of concerns applies at all levels in high-quality modern software—hiding code, hiding data, and hiding type. Each increases encapsulation and reduces coupling, which are essential for large-scale and robust software.