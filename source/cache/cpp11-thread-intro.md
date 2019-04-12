---
title: C++11 thread 标准库简介
date: 2019-04-02 00:23:56
tags:
    - cpp
---
简单介绍下c++11引入的`thread`标准库的简单用法。
![众人划桨开大船](/img/cpp11_threads_intro_title.jpg)
<!-- more -->

## 一个例子

我们可以使用矩形面积逼近的方法计算一元函数的定积分。由于三角函数$f(x) = \cos x$在区间$[0, 2\pi]$上的积分为$0$，便于查验结果的正确性。所以下面的代码都是计算这个问题。其中，$\Delta x = 2\pi/N$。
$$I = \int_{0}^{2\pi}\cos x dx = \sum_{i=1}^{N}\cos x_i\Delta x$$

我们先给出一个naive解法，顺序累加：
``` cpp
#include <cmath>
#include <chrono>
#include <iostream>

#define NOW() std::chrono::system_clock::now()
// 计算两个time point之间经过的毫秒数
#define TIMEIT(start) std::chrono::duration_cast<std::chrono::milliseconds>(\
    std::chrono::system_clock::now() - start).count()

// 这里N给了很大，利于多线程版本发挥
const int N = 10000000;
const double PI = acos(-1);

double integral(int n=N) {
    const double step = 2 * PI / (n-1);
    double sum = 0.;
    double x = 0.;
    for (int i = 0; i < n; ++i, x += step) {
        sum += cos(x) * step;
    }
    return sum;
}


int main() {
    auto start = NOW();
    double res = integral();
    auto duration = TIMEIT(start);
    std::cout << "res = " << res << "\n";
    std::cout << "time elapsed: " << duration << " ms" << std::endl;
    return 0;
}
```
在我的mbp上的测试结果为：

``` bash
res = 6.28803e-07
time elapsed: 117 ms
```

## 老师，能不能快一点？

下面引入多线程进行加速。我们可以将连加的`N`个数平均分成`num of threads`组，每个线程只计算自己对应的那个分组的矩形面积。为了避免在多线程中写某个变量，使用一个数组存储各个分组的结果。最后在主线程中将所有的中间结果相加起来，得到最后的结果。

在C++11中，可以使用`std::thread`获得线程资源。参考此页面[constructor of std::thread](https://en.cppreference.com/w/cpp/thread/thread/thread)，可以知道`std::thread`的构造函数如下：

``` cpp
template< class Function, class... Args > 
explicit thread( Function&& f, Args&&... args );
```

参数意义如下：
> f: Callable object to execute in the new thread
args...: arguments to pass to the new function

所以，我们需要把将要进行的计算放到一个可调用对象（callable object）中，将其作为构造函数的第一个参数，而实参作为后面的参数。

最简单地，我们可以使用函数名：

``` cpp
// 计算 \sum_{x=start}^{start+n*step} f(x) dx -> res
void integral(double start, int n, double step, double& res) {
    res = 0.;
    double x = start;
    for (int i = 0; i < n; ++i, x += step) {
        res += cos(x) * step;
    }
}

// 构造 thread
double res = 0.;
// 使用 std::ref 获得左值引用
std::thread t1(integral, 0., 100, 0.01, std::ref(res));
```

或者，可以使用函数对象（`Functor`），好处是我们可以将`res`放入函数对象内部。

``` cpp
class IntegralComputer {
public:
    void operator()(double start, int n, double step) {
        res_ = 0.;
        double x = start;
        for (int i = 0; i < n; ++i, x += step) {
            res_ += cos(x) * step;
        }
    }
private:
    double res_;
};

// 当然，如果真的这样写，我们就没法拿到参数中匿名函数对象里面的res了
std::thread t2(IntegralComputer(), 0., 100, 0.01);
```

还可以使用lambda匿名函数：

``` cpp
double res = 0.;
std::thread t3([&](double start, int n, double step) {
    res = 0.;
    double x = start;
    for (int i = 0; i < n; ++i, x += step) {
        res += cos(x) * step;
    }
}, 0., 100, 0.01);
```

完整的一个示例代码如下：

``` cpp
#include <cmath>
#include <chrono>
#include <iostream>
#include <thread>
#include <numeric>
#include <vector>

#define NOW() std::chrono::system_clock::now()
#define TIMEIT(start) std::chrono::duration_cast<std::chrono::milliseconds>(\
    std::chrono::system_clock::now() - start).count()

const int N = 10000000;
const double PI = acos(-1);
// 使用4个线程
const int NUM_THREAD = 4;

void integral(double start, int n, double step, double& res) {
    res = 0.;
    double x = start;
    for (int i = 0; i < n; ++i, x += step) {
        res += cos(x) * step;
    }
}

int main() {
    // 机器硬件能支持的最大线程数，供参考
    std::cout << "hint num thread: " << std::thread::hardware_concurrency() << std::endl;
    auto start = NOW();
    std::vector<std::thread> ts;
    double tmp[NUM_THREAD];
    const int every_n = N / NUM_THREAD;
    const double step = 2 * PI / (N - 1);
    for (int i = 0; i < NUM_THREAD; ++i) {
        // thread 一旦被构建，马上开始运行
        ts.emplace_back(std::thread(integral, i * every_n * step, N / NUM_THREAD, step, std::ref(tmp[i])));
    }
    // 使用 join 表示主进程将等待子进程完成
    // 由于每个进程任务量差不多，预计完成时间也差不多，所以我们依次将其join即可
    for (auto& t: ts) {
        t.join();
    }
    double res = std::accumulate(tmp, tmp + NUM_THREAD, 0.);
    auto duration = TIMEIT(start);
    std::cout << "res = " << res << "\n";
    std::cout << "time elapsed: " << duration << " ms" << std::endl;
    return 0;
}
```

## 竞争

设想一个银行账户系统，每个用户下面记录了他的银行账户余额。假设小明的账户余额为100元，小明和小兰分别在两个ATM对小明的账户进行操作，小明存入100元，小兰取出50元。我们可以知道，小明的账户余额应该是$100 + 100 - 50 = 150$元。在这个过程中，对小明账户余额的操作可以写成下面的流程：

```
=== 小明的部分
1. 取账户余额为 100
2. 在其上加上 100 -> 200
3. 将 200 写入账户余额
=== 小兰的部分
4. 取账户余额为 200
5. 在其上减去 50 -> 150
6. 将 150 写入账户余额 
```

然而，上面这种情况只考虑了两个人顺序操作。如果刚好在1步与4步之间，小兰就操作了她的ATM呢？那第四步取账户余额时，取出来的值就仍然是100，因为更新后的值200还没有被写入账户信息中。这样，当小兰执行完6之后，账户余额就被写入了50。

上面的例子中，ATM就是两个线程，账户余额是两个线程共享的变量。当没有任何额外保护的情况下，就可能会出现上面的情况，对数据的访问出现了竞争。

下面的例子中，为了突出上面的例子，我们让小明操作时，sleep掉2秒钟，以便小兰的操作有机可乘。
```
int g_x = 0;  // global x

void xiaoming() {
    // 先标记下 g_x
    int tmp = g_x;
    g_x = 1 - g_x;
    // sleep 2s，在这个时候，小兰修改了g_x的值，但是小明不知道
    std::this_thread::sleep_for(std::chrono::seconds(2));
    // 如果数据没问题，应该是 1 - tmp
    if (g_x != 1 - tmp) {
        std::cout << "g_x should be " << 1-tmp << ", but get " << g_x <<std::endl;
    } else {
        std::cout << "g_x check right." << std::endl;
    }
}
void xiaolan() {
    // just reset g_x
    g_x = 0;
}

// main
std::thread t1(xiaoming);
std::thread t2(xiaolan);
t1.join();
t2.join();

// 输出：
```

## 加个锁，这段时间我来独占资源

为了解决上面的问题，我们可以使用`mutex`进程锁。使用下面的方法在访问共享资源前后进行加锁和解锁，就能够保证在这期间，只有加锁的进程能够访问资源。
``` cpp
#include <mutex>
std::mutex g_mutex;
void fun() {
    // other part
    g_mutex.lock();
    // 处理银行账户余额
    g_mutex.unlock();
}
```