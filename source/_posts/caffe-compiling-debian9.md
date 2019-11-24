---
title: Debian9 编译Caffe的一个坑
date: 2019-11-24 19:12:48
tags:
---

记录一个编译Caffe的坑。环境，Debian 9 + GCC 6.3.0，出现的问题：

```
In file included from /usr/local/cuda/include/cuda_runtime.h:120:0,
                 from <command-line>:0:
/usr/local/cuda/include/crt/common_functions.h:74:24: error: token ""__CUDACC_VER__ is no longer supported.  Use __CUDACC_VER_MAJOR__, __CUDACC_VER_MINOR__, and __CUDACC_VER
_BUILD__ instead."" is not valid in preprocessor expressions
 #define __CUDACC_VER__ "__CUDACC_VER__ is no longer supported.  Use __CUDACC_VER_MAJOR__, __CUDACC_VER_MINOR__, and __CUDACC_VER_BUILD__ instead."
```

如果你和我一样，自从从Github clone Caffe后很长时间没有与master合并过，就有可能出现这个问题。

解决方法：这个问题应该是和boost有关，最初我看到的解决方法是将boost升级到1.65.1。不过感觉好麻烦，后来找到了这个[github issue](https://github.com/NVIDIA/caffe/issues/408)，修改`include/caffe/common.hpp`即可。

<!-- more -->

![diff修改](/img/fix_caffe_for_boost_CUDACC_VER_error.png)