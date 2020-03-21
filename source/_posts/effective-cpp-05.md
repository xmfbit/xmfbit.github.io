---
title: Effective CPP 阅读 - Chapter 5 实现
date: 2017-05-01 14:56:54
tags:
    - cpp
---
第四章中探讨了如何更好地提出类的定义和函数声明，精心设计的接口能让后续工作轻松不少。然而如何能够正确高效地实现，也是一件重要的事情。行百里者半九十。

<!-- more -->

## 26 尽可能延后变量定义式的出现时间
变量，尤其是自定义对象，一旦被定义，就会调用构造函数；一旦生命周期结束，又要调用析构函数。所以，延后变量定义的时间，并且最好能够给定恰当的初始值，这样能够提高代码执行效率。

## 27 尽量少做转型动作
第一，安全考虑。C++的类型系统保证类型错误不会发生。理论上如果你的代码“很干净”地通过编译，就表明它不意图在任何对象上执行不安全和无意义的操作，这是一个很有价值的保险。不要轻易打破。

第二，效率问题。这里展开说。

C++中的转型动作有以下三种：

- `(T)expression`，C时代的风格
- `T(expression)`，函数风格
- 新式风格，包括`static_cast`,`const_cast`,`dynamic_cast`和`reinterpret_cast`四种。

作者不提倡使用两种旧风格，而使用下面的四种。一种理由是它们容易被自动化代码检查工具匹配检查。

对于 `const_cast`，用于脱掉某对象的`const`属性。

对于`static_cast`，用于进行强制类型转换。例如将non-const类型转换为const类型，或者将`double`类型转换为`int`等。

对于`dynamic_cast`，用于执行安全向下转换，也就是用于决定某对象是否归属继承体系中的某个类型。注意，`dynamic_cast`的很多实现都很慢，尤其是继承深度较深时，也是这几个里面唯一可能造成重大（注意，并非其他三者不会带来执行时间开销）运行成本的动作。

很多时候，之所以使用`dynamic_cast`是因为你想在一个（你认为是）某个派生类对象中执行派生类中（并非从基类继承）的成员函数，但你的手上只有指向这个派生类对象的base指针或引用。这种情况下，也许将派生类的这个函数在基类中也定义一个空函数体的函数，再由派生类重写可能更好。

对于`reinterpret_cast`，它用来实现低级转型，实际动作和结果依赖于编译器，表示它不可移植。例如将指针转换为`int`，此转换是在bit层面的转换，更详细的信息可以参见[cpp reference的介绍](http://en.cppreference.com/w/cpp/language/reinterpret_cast)。

在它们之中，`reinterpret_cast`和`const_cast`完全是编译器层面的东西。`static_cast`会导致编译器生成对应CPU指令，但是是在编译器就能决定的。`dynamic_cast`是在运行时多态的一种手段。

## 28 避免返回指向对象内部成分的句柄（handle）
当成员函数返回类内部私有成员变量（或者私有成员函数，较少见）的句柄（如指针，引用或迭代器），而且成员变量的资源又存储于对象之外，这时候，虽然可以将此成员函数声明为`const`，但是实际上并不能避免通过此句柄修改资源的情况发生。

例如在自定义的`string`类中，使用堆上的数组储存字符串。如果某个成员函数能够返回字符串数组，那么可以使用这个指针修改数组内的值，而这也是符合`const`约定的。

所以，若无十分必要，不要返回对象内部的句柄。有此需要时，首先考虑是否应返回`const handle&`。

即使这样，还有可能造成返回的句柄比变量本身生命周期更长，也就是句柄所指之物已经被析构，句柄此时成为空悬状态，造成问题。

## 29 为“异常安全”努力是值得的
异常安全函数是指即使发生异常，也不会泄露资源或者允许任何数据结构被破坏。由于在代码执行过程中，可能发生内存申请失败等等异常，导致我们逻辑上已经设想好的程序控制流被中断，造成内存泄漏（后续的`delete`操作没有执行）。此外，我们希望如果异常发生，变量的值（程序的状态）能够恢复到异常发生之前。

让我们先看一个不满足异常安全的函数例子：
``` cpp
// 自定义`Menu`类中修改背景图片的成员函数
void Menu::changeBg(std::istream& imgSrc) {
    lock(&this->mutex);  // 互斥锁 1
    delete this->bg;     // 释放本已有的bg2
    ++this->imgChangeCnt;      // 计数器  3
    bg = new Image(imgSrc);    // 新图片  4
    unlock(&this->mutex);      // 释放锁  5
}
```

上面的代码存在以下问题，使得它不满足异常安全性。

- 资源泄漏。一旦第4行内存申请失败，那么第五行无法执行，互斥锁永远把持住了。
- 数据被破坏。还是上面的情况，则`bg`此时的资源已经被析构，而且计数器的值也增加了。·

解决第一个问题，可以考虑使用智能指针，即条款13中的使用对象管理资源。

我们将异常安全分为以下三类：

- 基本型。异常被抛出后，程序内的数据不会被破坏。但是并不保证程序的现实状态（究竟`bg`是何值）
- 强烈保证。异常抛出后，程序恢复到该函数调用前的状态。copy-and-swap策略是达成这一目标的常见方法。首先为待修改的对象原件做出一份副本，然后在副本上做一切修改。若有任何修改抛出异常，则原件不受影响。待所有修改完成后，再将修改过后的副本和原件在一个不抛出异常的swap操作中交换。

在这里，常常采用pImpl技术，也就是在对象中仅存储资源的（智能）指针，在swap中只操作该指针即可。

- 绝不抛出异常。作用于内置类型（`int`或指针等）身上的所有操作都提供了nothrow保证。

## 30 透彻了解内联的方方面面
`inline`函数在编译期实现函数本体的替换，避免了函数调用的开销，还可能使得编译器基于上下文进行优化，鼓励使用`inline`替换函数宏定义。

然而，`inline`不要乱用。首先，`inline`会使得目标码体积变大。可能造成额外的换页行为，降低高速缓存的命中概率，反而造成性能的损失。

另一方面，`inline`只是对编译器的申请，不是真的一定内联。

`inline`函数通常定义在头文件中（或者直接定义在类的内部，这样无需加入`inline`关键字），这是因为在编译中编译器需要知道这个函数具体长什么样子，才能够实现内联。

有时候，虽然编译器有意愿内敛某函数，但是还是会为它产生一个函数本体。这常常发生在取某个内联函数地址时。与此并提，编译器通常不对通过函数指针调用的内联函数进行内联。也就是说，是否真的内联，还与函数的调用方式有关。

作者给出的建议是，一开始不要将任何函数内联，之后使用profile工具进行优化。不要忘记28法则，80%的程序执行时间花在了20%的代码上。除非找对了目标，否则优化都是无用功。将内联函数应用于调用频繁且小型的函数身上。

## 31 将文件间的编译依存关系降至最低
C++的头文件包含机制饱受批评。连串的编译依存关系常常使得项目的编译时间大大加长。

首先，程序库头文件应该“完全且仅有声明式”的存在，将实现代码放入cpp文件中。

另外，之所以C++编译时容易出现“牵一发而动全身”的情况，是因为C++与Java等语言不同。在Java中编译器只分配一个指针指向实际对象，也就不需要知道对象的实际大小。而C++编译器却需要知道对象中每个成员变量的明确定义，才能知道对象的实际大小，从而在内存中分配空间。

从这里出发，我们可以参考Java等语言中的思路，建立一个handle类，在其中包含原来那个类的完全数据，而在新的类中定义一个指向该handle类的指针，这也就是前面所提到的pImpl方法。

使用这种思虑，定义的包含有`Date`类型对象（指明这个人的生日）的`Person`类如下：

``` cpp
#include <string>   // for string
#include <memory>   // for shared_ptr

class PersonImpl;   // Person实现类的前置声明
class Date;         // Person接口用到的类的前置声明

class Person {
public:
    Person(const std::string& name, const Date& birthday);
    std::string name() const;
private:
    std::shared_ptr<PersonImpl> pImpl;   // 指向实现类
};

/*****************实现文件****************/
#include "Person.h"
#include "PersonImpl.h"
Person::Person(const std::string& name, const Date& birthday):
    pImpl(new PersonImpl(name, birthday)) {}

std::string Person::name() const {
    return pImpl->name();
}
```

在上面的代码中，通过构造handle类`PersonImpl`，在`Person`中我们只需要前置声明`Date`，而无需包含头文件`date.hpp`。这样，即使`Date`或者`Person`有修改，影响也仅限于`Date`的实现文件和`PersonImpl`而已，不会传导到`Person`和使用了`Person`的其他代码文件。通过这种做法，实际上`Person`成为了一个单纯的接口，具体的实现在`PersonImpl`中完成，实现了“接口与实现的分离”。

综上：

- 如果使用object pointer或者object reference可以完成任务，就不要使用object。只要前置声明就可以定义出指向该类型的pointer或者reference，但是需要完整地定义式才能定义object。
- 如果能够，尽量用类的声明式替换定义式 。注意，当声明某个函数而它用到某个类时，你并不需要这个类的定义式。即使函数以pass-by-value方式传递参数（通常情况下这也不是一个好主意）或返回值。
- 为声明式和定义式提供不同的头文件（`Person`本身和`PersonImpl`）。这两个文件应该保持一致。声明式改变了，需要修改定义式头文件。程序库客户应该包含声明文件。

除了上面的方法，也可以将`Person`定义为抽象基类（Caffe中的`Layer`就是类似的模式）。为了达成这一目标，`Person`需要一个虚构造函数（见条款7）和一系列的纯虚函数（作为接口，等待派生类重写实现）。如下所示：

``` cpp
class Person {
public:
    virtual ~Person();
    virtual string name() const = 0;
};
```

客户必须能够为这种类创建对象。通常的做法是调用一个工厂函数，返回派生类的（智能）指针。这样的函数常常在抽象基类中声明为`static`。

``` cpp
class Person {
public:
    static shared_ptr<Person> create(const string& name, const Date& birthday);
// ... 刚才的其他代码
};
```

当然，要想使用，我们还必须定义派生类实现相应的接口。

``` cpp
class RealPerson: public Person {
public:
    RealPerson(const string& name, const Date& birthday): name(name), birthDate(birthday) {}
    virtual ~RealPerson() {}
    string name() const { return this->name; }
private:
    string name;
    Date birthDate;
};
```

上面的工厂函数`create()`的实现：

``` cpp
shared_ptr<Person> Person::create(const string& name, const Date& date) {
    return shared_ptr<Person>(new RealPerson(name, date));
}
```

实际应用中的工厂函数会像工厂一样，根据客户需要，产出不同的派生类对象。

当然，使用上述技术增大了程序运行时间开销和内存空间。这需要在工程中分情况讨论。是否这部分的开销大到了需要无视接口实现分离原则的地步？如果是的，那就用具象的类代替他们。但是，不要因噎废食。
