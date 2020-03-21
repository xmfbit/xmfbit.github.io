---
title: Caffe 中的 SyncedMem介绍
date: 2018-01-12 14:05:59
tags:
     - caffe
---
`Blob`是Caffe中的基本数据结构，类似于TensorFlow和PyTorch中的Tensor。图像读入后作为`Blob`，开始在各个`Layer`之间传递，最终得到输出。下面这张图展示了`Blob`和`Layer`之间的关系：
 <img src="/img/caffe_syncedmem_blob_flow.jpg" width = "300" height = "200" alt="blob的流动" align=center />

Caffe中的`Blob`在实现的时候，使用了`SyncedMem`管理内存，并在内存（Host）和显存（device）之间同步。这篇博客对Caffe中`SyncedMem`的实现做一总结。
<!-- more -->

## SyncedMem的作用
`Blob`是一个多维的数组，可以位于内存，也可以位于显存（当使用GPU时）。一方面，我们需要对底层的内存进行管理，包括何何时开辟内存空间。另一方面，我们的训练数据常常是首先由硬盘读取到内存中，而训练又经常使用GPU，最终结果的保存或可视化又要求数据重新传回内存，所以涉及到Host和Device内存的同步问题。

## 同步的实现思路
在`SyncedMem`的实现代码中，作者使用一个枚举量`head_`来标记当前的状态。如下所示：

``` cpp
// in SyncedMem
enum SyncedHead { UNINITIALIZED, HEAD_AT_CPU, HEAD_AT_GPU, SYNCED };
// 使用过Git吗？ 在Git中那个标志着repo最新版本状态的变量就叫 HEAD
// 这里也是一样，标志着最新的数据位于哪里
SyncedHead head_;
```

这样，利用`head_`变量，就可以构建一个状态转移图，在不同状态切换时进行必要的同步操作等。
![状态转换图](/img/caffe_syncedmem_transfer.png)

## 具体实现
`SyncedMem`的类声明如下：

``` cpp
/**
 * @brief Manages memory allocation and synchronization between the host (CPU)
 *        and device (GPU).
 *
 * TODO(dox): more thorough description.
 */
class SyncedMemory {
 public:
  SyncedMemory();
  explicit SyncedMemory(size_t size);
  ~SyncedMemory();
  // 获取CPU data指针
  const void* cpu_data();
  // 设置CPU data指针
  void set_cpu_data(void* data);
  // 获取GPU data指针
  const void* gpu_data();
  // 设置GPU data指针
  void set_gpu_data(void* data);
  // 获取CPU data指针，并在后续将改变指针所指向内存的值
  void* mutable_cpu_data();
  // 获取GPU data指针，并在后续将改变指针所指向内存的值
  void* mutable_gpu_data();
  // CPU 和 GPU的同步状态：未初始化，在CPU（未同步），在GPU（未同步），已同步
  enum SyncedHead { UNINITIALIZED, HEAD_AT_CPU, HEAD_AT_GPU, SYNCED };
  SyncedHead head() { return head_; }
  // 内存大小
  size_t size() { return size_; }

#ifndef CPU_ONLY
  void async_gpu_push(const cudaStream_t& stream);
#endif

 private:
  void check_device();

  void to_cpu();
  void to_gpu();
  void* cpu_ptr_;
  void* gpu_ptr_;
  size_t size_;
  SyncedHead head_;
  bool own_cpu_data_;
  bool cpu_malloc_use_cuda_;
  bool own_gpu_data_;
  // GPU设备编号
  int device_;

  DISABLE_COPY_AND_ASSIGN(SyncedMemory);
};  // class SyncedMemory
```

我们以`to_cpu()`为例，看一下如何在不同状态之间切换。

``` cpp
inline void SyncedMemory::to_gpu() {
  // 检查设备状态（使用条件编译，只在DEBUG中使能）
  check_device();
#ifndef CPU_ONLY
  switch (head_) {
  case UNINITIALIZED:
    // 还没有初始化呢~所以内存啥的还没开
    // 先在GPU上开块显存吧~
    CUDA_CHECK(cudaMalloc(&gpu_ptr_, size_));
    caffe_gpu_memset(size_, 0, gpu_ptr_);
    // 接着，改变状态标志
    head_ = HEAD_AT_GPU;
    own_gpu_data_ = true;
    break;
  case HEAD_AT_CPU:
    // 数据在CPU上~如果需要，先在显存上开内存
    if (gpu_ptr_ == NULL) {
      CUDA_CHECK(cudaMalloc(&gpu_ptr_, size_));
      own_gpu_data_ = true;
    }
    // 数据拷贝
    caffe_gpu_memcpy(size_, cpu_ptr_, gpu_ptr_);
    // 改变状态变量
    head_ = SYNCED;
    break;
  // 已经在GPU或者已经同步了，什么都不做
  case HEAD_AT_GPU:
  case SYNCED:
    break;
  }
#else
  // NO_GPU 是一个宏，打印FATAL ERROR日志信息
  // 编译选项没有开GPU支持，只能说 无可奉告
  NO_GPU;
#endif
}
```

注意到，除了`head_`以外，`SyncedMemory`中还有`own_gpu_data_`（同样，也有`own_cpu_data_`）的成员。这个变量是用来标志当前CPU或GPU上有没有分配内存，从而当我们使用`set_c/gpu_data`或析构函数被调用的时候，能够正确释放内存/显存的。



