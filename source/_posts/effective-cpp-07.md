---
title: Effective CPP 阅读 - Chapter 7 模板与泛型编程
date: 2017-06-23 20:19:07
tags:
    - cpp
---
模板是C++联邦中的重要成员。要想用好STL，必须了解模板。同时，模板元编程也是C++中的黑科技。
<!-- more -->

## 41 了解隐式接口和编译器多态
OOP总是以显式接口和运行期多态解决问题。通过虚函数，运行期将根据变量（指针或引用）的动态类型决定究竟调用哪一个函数。

而在模板与泛型世界中，例如下面的代码。没有明确指出，但是要求类型`T`支持操作符`<`。隐式接口不基于函数的声明，而是由有效表达式组成。

同时，模板的具现化（instantiated）是在编译期发生的。通过模板具现化和函数重载，实现了多态。
``` cpp
template <typename T>
bool fun(const T& a, const T& b) {return a < b;}
```

## 42 了解`typename`的双重意义
`typename`和`class`一样，都可以用来表明模板函数或者模板类。这时候两者完全相同，如下：
``` cpp
template <typename/class T>
void fun(T& a) {...}
```

但是有一个地方只能用`typename`，即标识嵌套从属类型名称。所谓从属类型名称，是指依赖于某个模板参数的名称。如果该类型名称呈嵌套状态，则称为嵌套从属名称。如：
``` cpp
// 打印容器内的第二个元素
template <typename C>
void print_2nd_element(const C& container) {
    // 嵌套类型
    typename C::const_iterator it(container.begin());
    cout << *++it;
}
```

不过不要在基类列或者成员初始化列表中以其作为基类的修饰符。如：

``` cpp
template <typename T>
class Derived: public Base<T>::Nested { //即使Nest是嵌套类型，这里也不用typename
public:
    explicit Derived(int x)
     :Base<T>::Nested(x) { // 成员初始化列表也不用
        typename Base<T>::Nested tmp; //这时候要使用
     }
};
```

## 43 学习处理模板化基类内的名称
这个问题的根源在于模板特化，造成特化版本与一般版本接口不同。因为编译器不能够在模板化的基类中寻找继承而来的名称。例如下面的离子：

``` cpp
class TypeA {
public:
    void fun();
};

class TypeB {
public:
    void fun();
};

template <typename Type>
class Base {
public:
    void do_something() {
        Type x;
        x.fun();
    }
};

template <typename Type>
class Derived: public Base<Type> {
public:
    void do_something_too() {
        // ...
        do_something();  // 调用基类的函数，这里无法编译
    }
};
```

这是因为存在如下可能，`Base<Tyep>`对某种`Type`进行了特化。
``` cpp
template <>
class Base<TypeC> {
public:
    // 这里没有实现 dom_somthing 函数
};
```

所以存在这样的可能：`class Derived<TypeC>: public Base<TypeC>`，而这里面是没有`do_something`函数的。

为了解决这个问题，有三种办法：

- 在基类调用方法前面加上`this->`。
``` cpp
void do_something_too() {
    // ...
    this->do_something();  // 调用基类的函数，这里无法编译
}
```

- 使用`using`声明式。虽然这里和条款33一样都是用了这一技术，但是目的是不一样的。条款33中是因为派生类名称掩盖了基类，而这里是因为编译器本身就不进入基类中进行查找。

``` cpp
using Base<Type>::do_something;
void do_something_too() {
    // ...
    this->do_something();  // 调用基类的函数，这里无法编译
}
```

- 明白指出被调用的函数位于基类中。这种方法最不推荐，因为如果被调用的是虚函数，上述的明确资格修饰符会关闭虚函数的运行时绑定行为。

``` cpp
void do_something_too() {
    // ...
    Base<Type>::do_something();  // 调用基类的函数，这里无法编译
}
```

## 44 将与参数无关的代码抽离模板
由于模板会具象化生成多个类或者多个函数，所以最好将与模板参数无关的代码抽离出去，防止代码膨胀造成程序体积变大和效率下降。

如下所示是一个$N$阶方阵，其中`n`是阶数。如果我们对每个不同阶数的矩阵都写一遍矩阵求逆操作，会造成代码膨胀。

``` cpp
template <typename T, size_t n>
class Matrix {
public:
    void invert();
};
```

一种可行的解决方案是提取出一个公共的基类用于实现矩阵转置。

``` cpp
template <typename T>
class MatrixBase {
protected:
    MatrixBase(size_t n, T* pMem) // 存储矩阵大小和指针
     :size(n), pData(pMem) {}
    void setDataPtr(T* ptr) {pData = ptr;} // 设置指针
    void invert();   // 实现求逆
private:
    size_t size;
    T* pData;
};
```

而矩阵类继承自刚才这个没有设定非类型参数的基类。我们这里使用`private`继承来显示新的矩阵派生类只是根据旧的基类实现，而不是想表示Is-a的关系。
``` cpp
template <typename T, size_t n>
class Matrix: private MatrixBase<T> {
public:
    Matrix(): MatrixBase<T>(n, 0), pData(new T[n*n]) {
        this->setDataPtr(pData.get()); // 将指针副本传给基类
    }
private:
    boost::scoped_array<T> pData;
};
```

然而这样改动并不一定比原来的效率更高。因为按原来的写法，常量`n`是个编译器常量，编译器可以通过常量的广传做优化。所以，实际使用时，还是要以profile为准。

上述的例子是由于非类型参数造成的代码膨胀，而类型参数有时也会出现这种问题。如有的平台上`int`和`long`有相同的二进制表述。那么`vector<int>`和`vector<long>`的成员函数可能完全相同，也会造成代码膨胀。

在很多平台上，不同类型的指针二进制表述是一样的，所以凡是模板中含有指针，如`vector<int*>, list<const int*>`等，往往应该对成员函数使用唯一的底层实现。例如，当你在操作某个成员函数而它操作的是一个强类型指针（即`T*`）时，你应该让它调用另一个无类型指针`void*`的函数，由后者完成实际工作。

## 45 运用成员函数模板接受所有兼容类型
使用场景一，我们可以将某个类的拷贝构造函数写成模板函数，使其能够接受兼容类型。比如对于智能指针，我们希望能够实现原始指针那种向上转型的能力。如下所示，基类指针能够指向基类和派生类。
``` cpp
Base* p = new Base;
Base* p = new Derived;
```

``` cpp
// 一个通用的智能指针模板
template <typename T>
class SmartPointer {
public:
    // 为了兼容类型，需要再引入一个模板参数 U
    template <typename U>
    SmartPointer(const SmartPointer<U>& other)
        // 这里可能会发生指针之间的隐式类型转换
       :ptr(other.get())  {...}
    T* get() const { return ptr; }
private:
    T* ptr;
};
```

成员函数模板还可以用来作为赋值操作。
``` cpp
template <typename T>
class shared_ptr {
public:
    ...
    // 接受任意兼容的shared_ptr赋值
    template <typename Y>
    shared_ptr& operator = (shared_ptr<Y> const& r);
    // 接受任意兼容的auto_ptr赋值
    template <typename Y>
    shared_ptr& operator = (auto_ptr<Y> const& r);
};
```

不过声明泛化版本的拷贝构造函数和赋值运算符，并不会阻止编译器为你生成默认的版本。所以如果你想控制拷贝或赋值的方方面面，必须同时声明泛化版本和普通版本。即：
``` cpp
shared_ptr& operator = (shared_ptr const& r);
```
## 46 需要类型转换时请为模板定义（friend）非成员函数
回顾条款24，在其中指出，只有非成员函数才有能力在所有实参身上实施隐式类型转换。当这一规则延伸到模板世界中时，情况又有不同。如下所示，我们将实数类`Rational`声明为模板。

``` cpp
template <typename T>
class Rational {
public:
    Rational(const T& numerator=0, const T& denominator=1);
};

template <typename T>
const Rational<T> operator*(const Rational<T>& lhs,
                            const Rational<T>& rhs)
{...}

Rational<int> onehalf(1, 2);
Rational<int> res = onehalf * 2;  // 改成模板后便会编译错误！
```

这是因为在进行模板类型推导时，并未将`2`进行隐式类型转换（否则，就是一个鸡生蛋蛋生鸡的问题了）。所以编译器没法找到这样的一个函数。

解决方法是将这个运算符重载函数声明为`Rational<T>`的友元函数。这样，在`onehalf`被声明时，`Rational<int>`类被具现化，则该友元函数也被声明出来了。

然而这时也只能通过编译而链接出错。因为无法找到函数的定义。解决方法是将函数体移动到类内部（即声明时即定义）。对于更复杂的函数，我们可以定义一个在模板类外部的辅助函数，而由这个友元函数去调用。
``` cpp
template <typename T> class Rational;   // 前向声明
template <typename T>
const Rational<T> doMultiply(const Rational<T>& lhs,
                             const Rational<T>& rhs) {};

template <typename T>
class Rational {
public:
    Rational(const T& numerator=0, const T& denominator=1) {
    }
    // ...
    friend const Rational<T> operator*(const Rational& lhs,
                                       const Rational& rhs)
    {return doMultiply(lhs, rhs); }
};
```

## 47 使用`trait`表现类型信息
STL中的`advance`函数可以将某个迭代器移动给定的距离。但是对于不同的迭代器，我们需要采用不同的策略。
- 输入迭代器。输入迭代器只能前向移动，每次一步，而且是只读一次，模仿的是输入文件的指针。例如`istream_iterator`。
- 输出迭代器。输出迭代器只能向前移动，每次一步，而且是只写一次，模仿的是输出文件的指针。例如`ostream_iterator`。
- 前向迭代器。只能向前移动，每次一步，可以读或写所指物一次以上。例如单向链表。
- 双向迭代器。可以向前向后移动，每次一步，可以读或写所指物一次以上，例如双向链表。
- 随机迭代器。可以随意跳转任意距离，例如`vector`或原始指针。

为了对它们进行分类，C++有对应的tag标签。
``` cpp
struct input_iterator_tag {};
struct output_iterator_tag {};
struct forward_iterator_tag: public input_iterator_tag {};
struct bidirectional_iterator_tag: public forward_iterator_tag {};
struct random_access_iterator_tag: public bidirectional_iterator_tag {};
```

所以我们可以在`advance`的代码中，对迭代器的类型进行判断，从而采取不同的操作。`trait`就是能够让你在编译器获得类型信息。

我们希望`trait`也能够应用于内建类型，所以直接类型内的嵌套信息这种方案被排除了。因为我们无法对内建类型，如原始指针塞进去这个类型信息（对用户自定义的类型倒是很简单）。STL采用的方案是将其放入模板及其特化版本中。STL中有好几个这样的`trait`（而且C++11加入了更多），其中针对迭代器的是`iterator_traits`。

为了实现这一功能，我们要在定义相应迭代器的时候，指明其类型（通常通过`typedef`来实现）。如队列的迭代器支持随机访问，则：
``` cpp
template <typename T>
class deque {
public:
    class iterator {
    public:
        typedef random_access_iterator_tag iterator_category;
        // ...
    }
};
```

这样，我们就能在`iterator_traits`内部通过访问迭代器的`iterator_category`来获得其类型信息啦~如下所示，`iterator_traits`只是鹦鹉学舌般地表现`IterT`说自己是什么。

``` cpp
template <typename IterT>
struct iterator_traits {
    typedef typename IterT::iterator_category iterator_category;
};
```

如何支持原始指针呢？用模板特化就好了~

``` cpp
template <typename T>
struct iterator_traits<T*> {
    typedef random_access_iterator_tag iterator_category;
};
```

总结起来，如何设计并实现一个`traits`呢？

- 确认若干你想要获取到的类型相关信息，例如本例中我们想要获得迭代器的分类（category）。
- 为该信息取一个名称，如`iterator_category`
- 提供一个模板和相关的特化版本，内含你想要提供的类型相关信息。

好了，下面我们可以实现`advance`了。
``` cpp
template <typename IterT, typename DistT>
void advance(IterT& iter, DistT d) {
    if(typeid(typename std::iterator_traits<IterT>::iterator_category
        == typeid(std::random_access_iterator_tag) {
        // ...
    }
    // ...
}
```

然而，为什么要将在编译期能确定的事情搞到运行时再确定呢？我们可以通过函数重载的方法实现编译期的`if-else`功能。

我们为不同类型的迭代器实现不同的移动方法。
``` cpp
template <typename IterT, typename DistT>
void doAdvance(IterT& iter, Dist d, std::random_access_iterator_tag) {
    iter += d;
}

// ...其他类型的迭代器对应的 doadvance

// 用advance函数包装这些重载函数
template <typename Iter, typename DistT>
void advance(IterT& iter, Dist d) {
    doAdvance(iter, d, typename std::iterator_traits<IterT>::iterator_category());
    // 注意 typename
    // 注意传入的是对象实例，所以要 iterator_category()
}
```
也就是说
- 首先建立一组重载函数或函数模板（真正干活的劳工），彼此之间的差异只在`trait`参数。
- 建立包装函数（包工头），调用上述劳工函数并传递`trait`信息。

## 48 认识模板元编程
模板元编程（Template Metaprogram， TMP）能够实现将计算前移到编译器，能够实现早期错误侦测（如科学计算上的量度单位是否正确）和更高的执行效率（MXNet利用模板实现懒惰求值，消除中间临时量）。

条款47介绍了选择分支结构如何借由`trait`实现。这里介绍循环由递归模板具现化实现的方法。

为了生成斐波那契数列，我们首先定义一个模板参数为`n`的模板类。然后指出其值可以递归地由模板具现化实现。并通过模板特化给出递归基。

``` cpp
template <unsigned n>
struct F {
    enum {value = n * F<n-1>::value };
};

template <>
struct F<0> {
    enum {value = 1 };
};
```

TMP博大精深，想要深入学习，还是要参考相关书籍。
