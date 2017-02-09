---
title: 在Ubuntu14.04构建Caffe
date: 2017-02-09 20:59:05
tags:
    - tool
---
Caffe作为较早的一款深度学习框架，很是流行。然而，由于依赖项众多，而且Jia Yangqing已经毕业，所以留下了不少的坑。这篇博客记录了我在一台操作系统为Ubuntu14.04.3的DELL游匣7559笔记本上编译Caffe的过程，主要是在编译python接口时遇到的import error问题的解决和找不到HDF5链接库的问题。

![caffe](/img/caffe_image.jpg)

<!-- more -->
## 修改Makefile.config
当从github上clone下来Caffe的源码之后，我们首先需要修改Makefile.config文件，自定义配置。下面是我的配置文件，主要修改了CUDNN部分和Anaconda Python部分。

``` bash
## Refer to http://caffe.berkeleyvision.org/installation.html
# Contributions simplifying and improving our build system are welcome!

# cuDNN acceleration switch (uncomment to build with cuDNN).
USE_CUDNN := 1    # 这里我们使用cudnn加速

# CPU-only switch (uncomment to build without GPU support).
# CPU_ONLY := 1

# uncomment to disable IO dependencies and corresponding data layers
# USE_OPENCV := 0
# USE_LEVELDB := 0
# USE_LMDB := 0

# uncomment to allow MDB_NOLOCK when reading LMDB files (only if necessary)
#	You should not set this flag if you will be reading LMDBs with any
#	possibility of simultaneous read and write
# ALLOW_LMDB_NOLOCK := 1

# Uncomment if you're using OpenCV 3
# OPENCV_VERSION := 3

# To customize your choice of compiler, uncomment and set the following.
# N.B. the default for Linux is g++ and the default for OSX is clang++
# CUSTOM_CXX := g++

# CUDA directory contains bin/ and lib/ directories that we need.
CUDA_DIR := /usr/local/cuda
# On Ubuntu 14.04, if cuda tools are installed via
# "sudo apt-get install nvidia-cuda-toolkit" then use this instead:
# CUDA_DIR := /usr

# CUDA architecture setting: going with all of them.
# For CUDA < 6.0, comment the *_50 lines for compatibility.
# 这里可以去掉sm_20和21，因为实在是已经太老了
# 如果保留的话，编译时nvcc会给出警告
CUDA_ARCH := -gencode arch=compute_30,code=sm_30 \
		-gencode arch=compute_35,code=sm_35 \
		-gencode arch=compute_50,code=sm_50 \
		-gencode arch=compute_50,code=compute_50

# BLAS choice:
# atlas for ATLAS (default)
# mkl for MKL
# open for OpenBlas
BLAS := atlas
# Custom (MKL/ATLAS/OpenBLAS) include and lib directories.
# Leave commented to accept the defaults for your choice of BLAS
# (which should work)!
# BLAS_INCLUDE := /path/to/your/blas
# BLAS_LIB := /path/to/your/blas

# Homebrew puts openblas in a directory that is not on the standard search path
# BLAS_INCLUDE := $(shell brew --prefix openblas)/include
# BLAS_LIB := $(shell brew --prefix openblas)/lib

# This is required only if you will compile the matlab interface.
# MATLAB directory should contain the mex binary in /bin.
# MATLAB_DIR := /usr/local
# MATLAB_DIR := /Applications/MATLAB_R2012b.app

# NOTE: this is required only if you will compile the python interface.
# We need to be able to find Python.h and numpy/arrayobject.h.
PYTHON_INCLUDE := /usr/include/python2.7 \
		/usr/lib/python2.7/dist-packages/numpy/core/include
# Anaconda Python distribution is quite popular. Include path:
# Verify anaconda location, sometimes it's in root.
# 这里我们使用Anaconda
ANACONDA_HOME := $(HOME)/anaconda2
 PYTHON_INCLUDE := $(ANACONDA_HOME)/include \
		 $(ANACONDA_HOME)/include/python2.7 \
		 $(ANACONDA_HOME)/lib/python2.7/site-packages/numpy/core/include

# Uncomment to use Python 3 (default is Python 2)
# PYTHON_LIBRARIES := boost_python3 python3.5m
# PYTHON_INCLUDE := /usr/include/python3.5m \
#                 /usr/lib/python3.5/dist-packages/numpy/core/include

# We need to be able to find libpythonX.X.so or .dylib.
#PYTHON_LIB := /usr/lib
 PYTHON_LIB := $(ANACONDA_HOME)/lib

# Homebrew installs numpy in a non standard path (keg only)
# PYTHON_INCLUDE += $(dir $(shell python -c 'import numpy.core; print(numpy.core.__file__)'))/include
# PYTHON_LIB += $(shell brew --prefix numpy)/lib

# Uncomment to support layers written in Python (will link against Python libs)
WITH_PYTHON_LAYER := 1

# Whatever else you find you need goes here.
INCLUDE_DIRS := $(PYTHON_INCLUDE) /usr/local/include
LIBRARY_DIRS := $(PYTHON_LIB) /usr/local/lib /usr/lib

# If Homebrew is installed at a non standard location (for example your home directory) and you use it for general dependencies
# INCLUDE_DIRS += $(shell brew --prefix)/include
# LIBRARY_DIRS += $(shell brew --prefix)/lib

# NCCL acceleration switch (uncomment to build with NCCL)
# https://github.com/NVIDIA/nccl (last tested version: v1.2.3-1+cuda8.0)
# USE_NCCL := 1

# Uncomment to use `pkg-config` to specify OpenCV library paths.
# (Usually not necessary -- OpenCV libraries are normally installed in one of the above $LIBRARY_DIRS.)
# USE_PKG_CONFIG := 1

# N.B. both build and distribute dirs are cleared on `make clean`
BUILD_DIR := build
DISTRIBUTE_DIR := distribute

# Uncomment for debugging. Does not work on OSX due to https://github.com/BVLC/caffe/issues/171
# DEBUG := 1

# The ID of the GPU that 'make runtest' will use to run unit tests.
TEST_GPUID := 0

# enable pretty build (comment to see full commands)
Q ?= @
```

对于CUDNN，从Nvidia官方网站上下载后，可不按照官方给定的方法安装。直接将`include`中的头文件放于`/usr/local/cuda-8.0/include`下，将`lib`中的库文件放于`/usr/loca/cuda-8.0/lib64`文件夹下即可。

## 构建
使用`make -j8`进行编译，并使用`make pycaffe`生成python接口。并在`.bashrc`中添加内容：
```
export PYTHONPATH=/path_to_caffe/python:$PYTHONPATH
```

结果在`import caffe`时出现问题如下：
```
ImportError: libcudnn.so.5: cannot open shared object file: No such file or directory
```
解决方法如下，详见GitHub issue[讨论](https://github.com/NVIDIA/DIGITS/issues/8)。
```
sudo ldconfig /usr/local/cuda/lib64
```

然而仍有问题，如下：
```
ImportError: No module named google.protobuf.internal
```
解决方法如下，详见G+ caffe-user group的[帖子](https://groups.google.com/forum/#!topic/caffe-users/9Q10WkpCGxs)。
```
pip install protobuf
```

不过仍然存在的问题是远程SSH登录时，不能在`ipython`环境下导入caffe，不知为何。

使用`make test; make runtest`进行测试，结果提示HDF5动态链接库出现问题，怀疑与Anaconda中的HDF5冲突有关。错误信息如下：

```
error while loading shared libraries: libhdf5_hl.so.10: cannot open shared object file: No such file or directory
```

解决方法为手动添加符号链接，详见GitHub[讨论帖](https://github.com/BVLC/caffe/issues/1463)。

```
cd /usr/lib/x86_64-linux-gnu
sudo ln -s libhdf5.so.7 libhdf5.so.10
sudo ln -s libhdf5_hl.so.7 libhdf5_hl.so.10
```

## 测试
首先，通过`make runtest`看是否全部test可以通过。其次，可以试运行`example`下的LeNet训练。
```
cd $CAFFE_ROOT
./data/mnist/get_mnist.sh
./examples/mnist/create_mnist.sh
./examples/mnist/train_lenet.sh
```
