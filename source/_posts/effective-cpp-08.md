---
title: Effective CPP 阅读 - Chapter 8 定制new和delete
date: 2017-07-03 19:47:43
tags:
    - cpp
---
手动管理内存，这既是C++的优点，也是C++中很容易出问题的地方。本章主要给出分配内存和归还时候的注意事项，主角是`operator new`和`operator delete`，配角是new_handler，它在当`operator new`无法满足客户内存需求时候被调用。

另外，`operator new`和`operator delete`只用于分配单一对象内存。对于数组，应使用`operator new[]`，并通过`operator delete[]`归还。除非特别指定，本章中的各项既适用于单一`operator new`，也适用于`operator new[]`。

最后，STL中容器使用的堆内存是由容器拥有的分配器对象（allocator objects）来管理的。本章不讨论。
![memory_leak_everywhere](/img/effectivecpp_08_memory_leak_everywherre.jpg)
<!-- more -->

## 49 了解new-handler的行为
什么是new-handler？当`operator new`无法满足内存分配需求时，会抛出异常。在抛出异常之前，会先调用一个客户指定的错误处理函数，这就是所谓的new-handler，也就是一个擦屁股的角色。

为了指定new-handler，必须调用位于标准库`<new>`的函数`set_new_handler`。其声明如下：
``` cpp
namespace std {
    typedef void (*new_handler) ();
    new_handler set_new_handler(new_handler p) throw();
}
```

其中，传入参数`p`是你要指定的那个擦屁股函数的指针，返回参数是被取代的那个原始处理函数。`throw()`表示该函数不抛出任何异常。

当`operator new`无法满足内存需求时，会不断调用`set_new_handler()`，直到找到足够的内存。更加具体的介绍见条款51.

一个设计良好的new_handler函数可以是以下的设计策略：

- 设法找到更多的内存可供使用，以便使得下一次的`operator new`成功。
- 安装另一个new_handler函数。即在其中再次调用`set_new_handler`，找到其他的擦屁股函数接盘。
- 卸载new_handler函数。即将`NULL`指针传进`set_new_handler()`中去。这样，`operator new`会抛出异常。
- 抛出`bad_alloc`（或其派生类）异常。
- 不返回（放弃治疗），直接告诉程序exit或abort。

有的时候想为不同的类定制不同的擦屁股函数。这时候，需要为每个类提供自己的`set_new_handler()`函数和`operator new`。如下所示，由于对类的不同对象而言，擦屁股机制都是相同的，所以我们将擦屁股函数声明为类内的静态成员。

``` cpp
class A {
public:
    static std::new_handler set_new_handler(std::new_handler p) throw();
    static void* operator new(std::size_t size) throw(std::bad_alloc);

private:
    static std::new_handler current_handler;
};

// 实现文件
std::new_handler A::set_new_handler(std::new_handler p) throw() {
    std::new_hanlder old = current_handler;
    current_handler = p;
    return old;
}
```
静态成员变量必须在类外进行定义（除非是`const`且为整数型），所以需要在类外定义：
``` cpp
// 实现文件
std::new_handler A::current_handler = 0;
```

在实现自定义的`operator new`的时候，首先调用`set_new_handler()`将自己的擦屁股函数安装为默认，然后调用global的`operator new`进行内存分配，最后恢复，把原来的擦屁股函数复原回去。书中，作者使用了一个类进行包装，利用类在scope的自动构造与析构，实现了自动化处理：

``` cpp
// 这个类实现了自动安装与恢复new_handler
class Helper {
public:
    explicit Helper(std::new_handler p): handler(p) {}
    ~Helper() {std::set_new_handler(handler); }
private:
    std::new_handler handler;
    // 禁止拷贝构造与赋值
    Helper(const Helper&);
    Helper& operator= (const Helper&);
};
// 实现类A自定义的operator new
void* A::operator new(std::size_t size) throw(std::bad_alloc) {
    // 存储了函数返回值，也就是原始的 new_handler
    Helper h(std::set_new_handler(current_handler));
    return ::operator new(size);
}
```

新的问题随之而来。如果我们想方便地复用上述代码呢？一个简单的方法是建立一个mixin风格的基类，这种基类用来让派生类继承某个唯一的能力（本例中是设定类的专属new_handler的能力）。而为了让不同的类获得不同的`current_handler`变量，我们把这个基类做成模板。

``` cpp
template <typename T>
class HandlerHelper {
public:
    static std::new_handler set_new_handler(std::new_handler p) throw();
    static void* operator new(std::size_t size) throw(std::bad_alloc);
    ... // 其他的new版本，见条款52
private:
    static std::new_handler current_handler;
};
// 实现部分的代码不写了，和上面的Helper和A中的对应内容基本完全一样
```

这样，我们只要让类`A`继承自`HandlerHelper<A>`即可（看上去很怪异。。。）：
``` cpp
class A: public HandlerHelper<A> {
    ...
};
```

## 50 了解替换`new`和`delete`的合适时机
最常见的理由（替换之后你能得到什么好处）：
- 检测运用上的错误。比如缓冲区越界，我们可以在`delete`的时候进行检查。
- 强化效能。编译器实现的`operator new`是为了普适性的功能，改成自定义版本可能提升效能。
- 收集使用上的统计数据。为了优化程序性能，理当先收集你的软件如何使用动态内存。自定义的`operator new`和`delete`能够收集到这些信息。

但是，写出能正常工作的`new`却不一定获得很好的性能。（各种细节上的问题，例如内存的对齐。也正因为如此，这里不再重复书上的一个具体实现）例如Boost库中的`Pool`对分配大量小型对象很有帮助。

## 51 编写`new`和`delete`时候需要遵守常规
自定义的`operator new`需要满足以下几点：
- 如果有足够的内存，则返回其指针；否则，遵循条款49的约定。
- 具体地，如果内存不足，那么应该循环调用new_handling函数（里面可能会清理出一些内存以供使用）。只有当指向new_handling的指针为`NULL`时，才抛出异常`bad_alloc`。
- C++规定，即使用户申请的内存大小为0，也要返回一个合法指针。这个看似诡异的行为是为了简化语言的其他部分。
- 还要避免掩盖正常的`operator new`。

下面就是一个自定义`operator new`的例子：
``` cpp
void* operator new(size_t size) throw(bad_alloc) {
    // 你的operator new也可能接受额外参数
    using namespace std;
    if(size == 0) {
        size = 1; // 处理0byte申请
    }
    while(true) {
        // ... try to alloc memory
        if(success) {
            return the pointer;
        }
        // 处理分配失败，找出当前的handler
        // 我们没有诸如get_new_handler()的方法来获取new_handler函数句柄
        // 所以只能用下面这种方法，利用set_new_handler的返回值获取当前处理函数
        new_handler globalHandler = set_new_handler(0);
        set_new_handler(globalHandler);

        if(globalHandler) {
            (*globalHandler)();
        } else {
            throw bad_alloc();
        }
    }
}
```

在自定义`operator delete`时候，注意处理空指针的情况。C++确保delete NULL pointer是永远安全的。
``` cpp
void operator delete(void* memory) throw() {
    if(memory == 0) return;
    // ...
}
```

## 52 写了placement new也要写placement delete
如果`operator new`接受的参数除了一定会有的那个`size_t`之外还有其他参数，那么它就叫做placement new。一个特别有用的placement new的用法是接受一个指针指向对象该被构造之处。声明如下所示：
``` cpp
void* operator new(size_t size, void* memory) throw();

```

上述placement new已经被纳入C++规范（可以在头文件`<new>`中找到它。）这个函数常用来在`vector`的未使用空间上构造对象。实际上这是placement的得来：特定位置上的new。有的时候，人们谈论placement new时，实际是在专指这个函数。

本条款主要探讨与placement new使用不当相关的内存泄漏问题。
当你写一个`new`表达式时，共有两个函数被调用：
- 分配内存的`operator new`
- 该类的构造函数

假设第一个函数调用成功，第二个函数却抛出异常。这时候我们需要将第一步申请得到的内存返还并恢复旧观，否则就会造成内存泄漏。具体来说，系统会调用和刚才申请内存的`operator new`对应的delete版本。

如果目前面对的是正常签名的`operator new delete`，不会有问题。不过若是当时调用的是修改过签名形式的placement new时，就可能出现问题。例如，我们有下面的placement new，它的功能是在分配内存的时候做一些logging工作。
``` cpp
// 某个类Wedget内部有自定义的placement new如下
static void* operator new(size_t size, ostream& logStream) throw (bad_alloc);

Widget* pw = new (std::cerr) Widget;
```
如果系统找不到相应的placement delete版本，就会什么都不做。这样，就无法归还已经申请的内存，造成内存泄漏。所以有必要声明一个placement delete，对应那个有logging功能的placement new。
``` cpp
static void operator delete(void* memory, ostream& logStream) throw();
// 这样，即使下式抛出异常，也能正确处理
Widget* pw = new (std::cerr) Widget;
```

然而，如果什么异常都没有抛出，而客户又使用了下面的表达式返还内存：
``` cpp
delete pw;
```
那么它调用的是正常版本的delete。所以，除了相对应的placement delete，还有必要同时提供正常版本的delete。前者为了解决构造过程中有异常抛出的情况，后者处理无异常抛出。

一个比较简单的做法是，建立一个基类，其中有所有正常形式的new和delete。
``` cpp
class StdNewDeleteForms {
public:
    // 正常的new和delete
    static void* operator new(std::size_t size) throw std::bad_alloc) {
        return ::operator new(size);
    }

    static void operator delete(void* memory) throw() {
        ::operator delete(memory);
    }
    // placement new 和 delete
    static void* operator new(std::size_t size, void* p) throw() {
        ::operator new(size, p);
    }
    static void operator delete(void* memory, void* p) throw() {
        ::operator delete(memory, p);
    }
    // nothrow new 和 delete
    static void* operator new(std::size_t size, const std::nothrow_t& nt) throw() {
        return ::operator new(size, nt);
    }
    static void operator delete(void* memory, const std::nothrow_t&) throw() {
        ::operator delete(mempry);
    }
};
```

上面这个类中包含了C++标准中已经规定好的三种形式的new和delete。那么，凡是想以自定义方式扩充标准形式，可利用继承机制和`using`声明（见条款39），取得标准形式。
``` cpp
class Widget: public StdNewDeleteForms {
public:
    // 使用标准new 和 delete
    using StdNewDeleteForms::operator new;
    using StdNetDeleteForms::operator delete;
    // 添加自定义的placement new 和 delete
    static void* operator new(std::size_t size,
        std::ostream& logStream) throw(std::bad_alloc);
    static void operator delete(void* memory, std::ostream& logStream) throw();
};
```
