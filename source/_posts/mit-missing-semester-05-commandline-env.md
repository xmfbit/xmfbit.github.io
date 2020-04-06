layout: post
title: MIT Missing Semester - Command-line Environment
date: 2020-04-06 18:52:29
tags:
    - tool
---

这是[MIT Missing Semester系列](https://missing.csail.mit.edu/2020/command-line/)的第五讲，主要关于shell下对于进程（Process)的控制。

<!-- more -->

# Job Control

## Kill Process

最简单的，要想结束当前正在进行的进程，使用`Ctrl+c`即可。而实际上，这是通过向其发送了`SIGINT`（INT是interrupt的意思）的信号量实现的。例如，执行如下脚本时候，如果按下`Ctrl+c`，并不会退出，而只是会执行`handler`函数。更多信号量的文档可以参考`man signal`。

``` py
#!/usr/bin/env python
import signal, time

def handler(signum, time):
    # 中断会进这里
    print("\nI got a SIGINT, but I am not stopping")

signal.signal(signal.SIGINT, handler)
i = 0
while True:
    time.sleep(.1)
    print("\r{}".format(i), end="")
    i += 1
```

## 休眠 (suspend)

使用`Ctrl-z`发送`SIGSTOP`信号可以使得进程暂时休眠，并可以后续继续运行。我们可以用下面的代码做实验。在`Ctrl+z`后，程序暂时休眠停止运行，计数器的值也不再更新，直到使用`fg`命令，才重新开始。

``` py
# 每隔1s打印当前计数器的值
import time
from itertools import count

def main():
    for i in count(0):
        print('current time: {}'.format(i))
        time.sleep(1)

if __name__ == '__main__':
    main()
```

![运行实例](/img/mit_missing_semester_05_example_suspend_process.png)

除了`fg`以外，还可以使用`bg`来开始被休眠的进程。只不过`bg`会让重新开始的进程在后台运行。

这里直接贴出讲师的一个例子：

![job control](/img/mit_missing_semester_05_example_job_control.png)

需要注意的是：

- `bg`的使用：将进程状态从suspending转到running
- `jobs`可以列出当前所有未完成的进程
- 可以使用`%number`的形式引用`jobs`列出的进程
- 除了快捷键，也可以使用`kill`向指定进程发送信号量，具体可以参考`man kill`

# SSH

远程ssh到开发机是常见的操作，这里有一些比较零散的知识点记录。

- 使用`ssh-copy-id $host_name`将当前机器的ssh public key拷贝到给定的远程host机器（之前我都是通过手动copy）
- 将文件拷贝到远程机器的N种方法：
   - 使用`ssh + tee`：`cat local_file | ssh remote_server tee server_file`，这是利用了`ssh remote_server`可以后接shell命令
   - 使用`scp`，这也是我最常用的命令了
   - 使用`rsync`，更强大的`scp`，可以跳过重复文件等
- 端口转发，例如在远程服务器的`8888`端口启动了jupyter notebook，可以使用`ssh -L 9999:localhost:8888 foobar@remote_server`将其转发到本机的`9999`端口，这样在本机浏览`localhost:9999`即可访问笔记本

![local port forwarding](/img/mit_missing_semester_05_local_port_forward.png)
![remote port forwarding](/img/mit_missing_semester_05_remote_port_forward.png)

# Exercise

## 创建alias

``` sh
alias dc="cd"
# 列出最常用的10个命令
## 将第一列（表示序号）设为空字符串
history | awk '{$1="";print substr($0,2)}' | sort | uniq -c | sort -n | tail -n 10
```

