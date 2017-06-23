---
title: Effective CPP 阅读 - Chapter 6 继承与面向对象设计
date: 2017-06-17 10:40:19
tags:
    - cpp
---
C++中允许多重继承，并且可以指定继承是否是public or private等。成员函数也可以是虚函数或者非虚函数。如何在OOP这一C++联邦中的重要一员的规则下，写出易于拓展，易于维护且高效的代码？
<!-- more -->

## 32 确定`public`继承塑模出Is-a的关系
请把这条规则记在心中：`public`继承意味着Is-a（XX是X的一种）的关系。适用于base class上的东西也一定能够用在derived class身上。因为每一个derived class对象也是一个base class的对象。

不过，在实际使用时，可能并不是那么简单。举个例子，在鸟类这个基类中定义了`fly()`这一虚函数，而企鹅很显然是一种鸟，但是却没有飞翔的能力。类似的情况需要在编程实践中灵活处理。

## 33 避免遮掩继承而来的名称
这个题材实际和作用域有关。当C++遇到某个名称时，会首先在local域中寻找，如果找到，就不再继续寻找。这样，derived class中的名称可能会遮盖base class中的名称。

一种解决办法是使用`using`声明。如下所示：
``` cpp
class Base {
public:
    virtual void f1() = 0;
    void f3();
    void f3(double);
};

class Derived: public Base {
public:
    using Base::f3;
    virtual void f1();
    void f3();
};

Derived d;
d.f1();  // 没问题，调用了Derived中的f1
d.f3();  // 没问题，调用了Derived中的f3
double x;
d.f3(x);  // 没问题，调用了Base中的f3。
// 但是如果没有using声明的话，Base::f3会被冲掉。
```

## 34 区分接口继承和实现继承
表面上直截了当的`public`继承，可以细分为函数接口继承和函数实现继承。以下面的这个例子来说明：

``` cpp
class Shape {
public:
    virtual void draw() const = 0;
    virtual void error(const std::string& msg);
    int getID() const;
};
class Rect: public Shape {...};
class Circle: public Shape {...};
```

- 纯虚函数
声明纯虚函数（如`draw()`函数）是为了让derived class只继承函数接口。乃是一种约定：“你一定要实现某某，但是我不管你如何实现”。
不过，你仍然可以给纯虚函数提供函数定义。

- 虚函数
非纯虚函数（如`error()`函数）的目的是，让derived class继承该函数的接口和缺省实现。乃是约定“你必须支持XX，但是如果你不想自己实现，可以用我提供的这个”。
然而可能会出现这样一种局面：derived class的表现与base class不同，但是又忘记了重写这个虚函数。为了避免这种情况，可以使用下面的技术来达到“除非你明确要求，否则我才不给你提供那个缺省定义”的目的。

``` cpp
class Base {
public:
    virtual void fun() = 0;   // 注意，我们改写成了纯虚函数
protected:
    void default_fun() {...};   // 缺省实现
};
// 此时，若想使用缺省实现，就必须显式地调用
class Derived: public Base {
public:
    virtual void fun() {
        default_fun();
    }
};
```
不过这样导致一个多余的`default_fun()`函数。如果不想添加额外的函数，我们可以使用上述提到的拥有定义的纯虚函数来实现。

``` cpp
class Base {
public:
    virtual void fun() = 0;
};
// 为纯虚函数提供定义
void Base::fun() {
    // 缺省行为
}

class Derived: public Base {
public:
    // 显式调用基类的纯虚函数，实现缺省行为
    virtual void fun() {Base::fun(); }
};
class Derived2: public Base {
public:
    // 实现自定义的行为
    virtual void fun() {
        // ...
    }
};
```

- 非虚函数
这意味着你不应该在derived class中定义不同的行为（老老实实用我给你的！），使得其继承了一份接口和强制实现。

## 35 考虑`virtual`函数之外的其他选择
虚函数使得多态成为可能。不过在一些情况下，为了实现多态，不一定非要使用虚函数。本条款介绍了一些相关技术。

在某游戏中，需要设计一个计算角色剩余血量的函数。下面是一种惯常的设计。
``` cpp
class GameCharacter {
public:
    virtual int healthValue() const;
};
```

- 使用non-virtual interface实现template method模式
这种流派主张`virtual`函数应该几乎总是私有的。较好的设计时将`healthValue()`函数设为非虚函数，并调用虚函数进行实现。这个调用函数中，可以做一些预先准备（互斥锁，日志等），后续可以做一些打扫工作。

``` cpp
class GameCharacter {
public:
    int healthValue() const {
        // ... 前期准备
        int ret_val = doHealthValue();
        // ... 后续清理
        return ret_val
    }
private:
    virtual int doHealthValue() const {...}
};
```
这样做的好处是基类明确定义了该如何实现求血量这个行为，同时又给了一定的自由，派生类可以重写`doHealthValue()`函数，针对自身的特点计算血量。

- 使用函数指针实现策略模式
上述方案实际上是对虚函数的调用进行了一次包装。我们还可以借由函数指针实现策略模式，为不同的派生类甚至不同的对象实例做出不同的实现。

``` cpp
class GameCharacter;   // 前置声明
// 计算血量的缺省方法
int defaultHealthValue(const GameCharacter&);

class GameCharacter {
public:
    typedef int (*HealthCalcFun) (const GameCharacter&);

    explicit GameCharacter(HealthCalcFun f=defaultHealthValue
        :healthFunc(f){
        // ...
    }
private:
    HealthCalcFun healthFunc;
};
```

这样，我们通过在构造时候传入相应的函数指针，就可以实现计算血量的个性化设置。比如两个同样的boss，血量下降方式就可以不一样。
或者我们可以在运行时候，通过设定`healthFunc`，来实现动态血量计算方法的变化。

- 借由`std::function`实现策略模式
作为上面的改进，我们可以使用`std::function`（C++11），这样，不止函数指针可以使用，函数对象等也都可以了。（关于`std::function`的大致介绍，可以看[这里](http://en.cppreference.com/w/cpp/utility/functional/function)）。

我们只需将上面的`typedef`改掉即可。不再使用函数指针，而是更加高级更加通用的`std::function`。

``` cpp
typedef std::function<int(const GameCharacter&)> HealthCalcFun;
```

- 使用古典的策略模式
如下图所示。对于血量计算，我们单独抻出来一个基类，并有不同的实现。`GameCharacter`类中则含有一个指向`HealthCalcFun`类实例的指针。

![使用UML表示的策略模式](/img/effectivecpp_strategy_pattern.png)

``` cpp
//我们首先定义HealthCalcFunc基类
class GameCharacter;    // 前向声明
class HealthCalcFunc {
public:
    virtual int calc(const GameCharacter& gc) const {...}
};

HealthCalcFunc defaultCalcFunc;

class GameCharacter {
private:
    HealthCalcFunc* pfun;
public:
    explicit GameCharacter(HealthCalcFunc* p=&defaultCalcFunc):
        pfun(p) {}
    int healthValue() const {
        return pfun->calc(*this);
    }
};
```

该条款给出了虚函数的若干替代方案。

## 36 绝不重新定义继承而来的非虚函数
在条款34中已经指出，非虚函数是一种实现继承的约定。派生类不应该重新定义非虚函数。这破坏了约定。

如下所示。
``` cpp
class B {
public:
    void mf() {...}
};
class D: public B {
public:
    void mf() {...}
};

D d;
B* pb = &d;
D* pd = &d;

pb->mf();   // 调用的是B::mf()
pd->mf();   // 调用的是D::mf()
```

这是因为非虚函数的绑定是编译期行为（和虚函数的动态绑定相对，其发生在运行时）。由于`pb`被声明为一个指向`B`的指针，所以其调用的是`B`的成员函数`mf()`。

为了不至于让自己陷入精神分裂与背信弃义的境地，请不要重新定义继承而来的非虚函数。

## 37 绝不重新定义继承而来的缺省参数值
由于条款36的分析，所以我们只讨论继承而来的是带有缺省参数的虚函数。这样一来，本条款背后的逻辑就很清晰了：因为缺省参数同样是静态绑定的，而虚函数却是动态绑定。让我们再解释一下。

静态类型是指在程序中被声明时的类型（不论其真实指向是什么）。
``` cpp
// Circle是Shape的派生类
Shape* ps;
Shape* pc = new Circle;   // 静态类型都是Shape
```

动态类型是指当前所指对象的类型。就上例来说，`pc`的动态类型是`Circle*`，而`ps`没有动态类型，因为它并没有指向任何对象实例。动态类型常常可以通过赋值改变。
``` cpp
ps = new Circle;   // 现在ps的动态类型是Circle*
```

虚函数是运行时决定的，取决于发出调用的那个对象的动态类型。

不过遵守此项条款，有时又会造成不便。看下例：

``` cpp
class Shape {
public:
    enum ShapeColor {RED, GREEN};
    virtual void draw(ShapeColor c=RED) const=0;
};

class Circle: public Shape {
public:
    virtual void draw(ShapeColor c=RED) const;
};
```

第一个问题，代码重复，我写了两遍缺省参数。第二造成了代码依存。比如我想换成`GREEN`为默认参数，需要在基类和派生类中同时修改。

一种解决方法是采用条款35中的替代设计，如NVI方法。令基类中的一个public的非虚函数调用私有的虚函数，而后者可以被派生类重新定义。我们只需要在public的非虚函数中定义缺省参数即可。

``` cpp
class Shape {
public:
    void draw(ShapeColor c=RED) const {
        doDraw(c);  // 调用私有的虚函数
    }
private:
    //真正的工作在此完成
    virtual void doDraw(ShapeColor c) const = 0;  
};

class Circle: public Shape {
private:
    virtual void doDraw(ShapeColor c) const;  // 派生类重写这个真正的实现
};
```

## 38 通过复合塑模has-a或“根据某物实现出”
复合是指某种对象内含其他对象。复合实际有两层意义，一种较好理解，即has-a，如人有名字、性别等他类，一种是指根据某物实现（is-implemented-in-terms-of）。例如实现消息管理的某个类中含有队列作为实现。

## 39 明智而审慎地使用`private`继承
私有继承意味着条款38中的“根据某物实现出”。例如`D`私有继承自`B`，不是说`D`是某种`B`，私有继承完全是一种技术上的实现（和对现实的抽象没有半毛钱关系）。`B`的每样东西在`D`中都是不可见的，也就是成了黑箱，因为它们本身就是实现细节，你只是考虑用`B`来实现`D`的功能而已。

但是复合也能达到相同的效果啊~我在`D`中加入一个`B`的对象实例不就好了？很多情况下的确是这样，如果没有必要，不建议使用私有继承。

## 40 明智而审慎地使用多重继承
使用多重继承有可能造成歧义。例如，`C`继承自`A`和`B`，而两个基类中都含有成员函数`mf()`。那么当`d.mf()`的时候，究竟是在调用哪个呢？你必须明确地指出,`d.A::mf()`。

使用多重继承还可能会造成“钻石型”继承。任何时候继承体系中某个基类和派生类之间有一条以上的相通路线，就面临一个问题，是否要让基类中的每个成员变量经由每一条路线被复制？如果只想保留一份，那么需要将`File`定为虚基类，所有直接继承自它的类采用虚继承。
![钻石型继承](/img/effectivecpp_diamond.png)

``` cpp
class File {...};
class InputFile: virtual public File {...};
class OutputFile: virtual public File {...};
class IOFile: public InputFile, public OutputFile {...};
```

从正确的角度看，public的继承总应该是virtual的。不过这样会造成代码体积的膨胀和执行效率的下降。

所以，如无必要，不要使用虚继承。即使使用，尽可能避免在其中放置数据（类似Java或C#中的接口Interface）

## 附注 `std::function`的基本使用
`std::function`的作用类似于函数指针，但是能力更加强大。我们可以将函数指针，函数对象，lambda表达式或者类中的成员函数作为`std::function`。
如下所示：

``` cpp
#include <functional>
#include <iostream>

struct Foo {
    Foo(int num) : num_(num) {}
    void print_add(int i) const { std::cout << num_+i << '\n'; }
    int num_;
};

void print_num(int i)
{
    std::cout << i << '\n';
}

struct PrintNum {
    void operator()(int i) const
    {
        std::cout << i << '\n';
    }
};

int main()
{
    // store a free function
    // 函数指针
    std::function<void(int)> f_display = print_num;
    f_display(-9);

    // lambda表达式
    // store a lambda
    std::function<void()> f_display_42 = []() { print_num(42); };
    f_display_42();

    // store the result of a call to std::bind
    // 绑定之后的函数对象
    std::function<void()> f_display_31337 = std::bind(print_num, 31337);
    f_display_31337();

    // store a call to a member function
    // 类中的成员函数，第一个参数为类实例的const reference
    std::function<void(const Foo&, int)> f_add_display = &Foo::print_add;
    const Foo foo(314159);
    f_add_display(foo, 1);
    f_add_display(314159, 1);

    // store a call to a data member accessor
    std::function<int(Foo const&)> f_num = &Foo::num_;
    std::cout << "num_: " << f_num(foo) << '\n';

    // store a call to a member function and object
    using std::placeholders::_1;
    std::function<void(int)> f_add_display2 = std::bind( &Foo::print_add, foo, _1 );
    f_add_display2(2);

    // store a call to a member function and object ptr
    std::function<void(int)> f_add_display3 = std::bind( &Foo::print_add, &foo, _1 );
    f_add_display3(3);

    // store a call to a function object
    std::function<void(int)> f_display_obj = PrintNum();
    f_display_obj(18);
}
```
