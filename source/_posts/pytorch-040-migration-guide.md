---
title: （译）PyTorch 0.4.0 Migration Guide
date: 2018-04-27 10:49:31
tags:
     - pytorch
---
PyTorch在前两天官方发布了0.4.0版本。这个版本与之前相比，API发生了较大的变化，所以官方也出了一个[转换指导](http://pytorch.org/2018/04/22/0_4_0-migration-guide.html)，这篇博客是这篇指导的中文翻译版。归结起来，对我们代码影响最大的地方主要有：

- `Tensor`和`Variable`合并，`autograd`的机制有所不同，变得更简单，使用`requires_grad`和上下文相关环境管理。
- Numpy风格的`Tensor`构建。
- 提出了`device`，更简单地在cpu和gpu中移动数据。

<!-- more -->
## 概述
在0.4.0版本中，PyTorch引入了许多令人兴奋的新特性和bug fixes。为了方便以前版本的使用者转换到新的版本，我们编写了此指导，主要包括以下几个重要的方面：

- `Tensors` 和 `Variables` 已经merge到一起了
- 支持0维的Tensor（即标量scalar）
- 弃用了 `volatile` 标志
- `dtypes`, `devices`, 和 Numpy 风格的 Tensor构造函数
- （更好地编写）设备无关代码

下面分条介绍。

## `Tensor` 和 `Variable` 合并
在PyTorch以前的版本中，`Tensor`类似于`numpy`中的`ndarray`，只是对多维数组的抽象。为了能够使用自动求导机制，必须使用`Variable`对其进行包装。而现在，这两个东西已经完全合并成一个了，以前`Variable`的使用情境都可以使用`Tensor`。所以以前训练的时候总要额外写的warpping语句用不到了。

``` py
for data, target in data_loader:
    ## 用不到了
    data, target = Variable(data), Variable(target)
    loss = criterion(model(data), target)
```
### `Tensor`的类型`type()`
以前我们可以使用`type()`获取`Tensor`的data type（FloatTensor，LongTensor等）。现在需要使用`x.type()`获取类型或`isinstance()`判别类型。

``` py
>>> x = torch.DoubleTensor([1, 1, 1])
>>> print(type(x))  # 曾经会给出 torch.DoubleTensor
"<class 'torch.Tensor'>"
>>> print(x.type())  # OK: 'torch.DoubleTensor'
'torch.DoubleTensor'
>>> print(isinstance(x, torch.DoubleTensor))  # OK: True
True
```

### `autograd`现在如何追踪计算图的历史
`Tensor`和`Variable`的合并，简化了计算图的构建，具体规则见本条和以下几条说明。

`requires_grad`, 这个`autograd`中的核心标志量,现在成了`Tensor`的属性。之前的`Variable`使用规则可以同样应用于`Tensor`，`autograd`自动跟踪那些至少有一个input的`requires_grad==True`的计算节点构成的图。

``` py
>>> x = torch.ones(1)  ## 默认requires_grad = False
>>> x.requires_grad
False
>>> y = torch.ones(1)  ## 同样，y的requires_grad标志也是False
>>> z = x + y
>>> ## 所有的输入节点都不要求梯度，所以z的requires_grad也是False
>>> z.requires_grad
False
>>> ## 所以如果试图对z做梯度反传，会抛出Error
>>> z.backward()
RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn
>>>
>>> ## 通过手动指定的方式创建 requires_grad=True 的Tensor
>>> w = torch.ones(1, requires_grad=True)
>>> w.requires_grad
True
>>> ## 把它和之前requires_grad=False的节点相加，得到输出
>>> total = w + z
>>> ## 由于w需要梯度，所以total也需要
>>> total.requires_grad
True
>>> ## 可以做bp
>>> total.backward()
>>> w.grad
tensor([ 1.])
>>> ## 不用有时间浪费在求取 x y z的梯度上，因为它们没有 require grad，它们的grad == None
>>> z.grad == x.grad == y.grad == None
True
```

### 操作 `requires_grad` 标志
除了直接设置这个属性，你可以使用`my_tensor.requires_grad_()`就地修改这个标志（还记得吗，以`_`结尾的方法名表示in-place的操作）。或者就在构造的时候传入此参数。

``` py
>>> existing_tensor.requires_grad_()
>>> existing_tensor.requires_grad
True
>>> my_tensor = torch.zeros(3, 4, requires_grad=True)
>>> my_tensor.requires_grad
True
```

### `.data`怎么办？What about .data?
原来版本中，对于某个`Variable`，我们可以通过`x.data`的方式获取其包装的`Tensor`。现在两者已经merge到了一起，如果你调用`y = x.data`仍然和以前相似，`y`现在会共享`x`的data，并与`x`的计算历史无关，且其`requires_grad`标志为`False`。

然而，`.data`有的时候可能会成为代码中不安全的一个点。对`x.data`的任何带动都不会被`aotograd`跟踪。所以，当做反传的时候，计算的梯度可能会不对，一种更安全的替代方法是调用`x.detach()`，仍然会返回一个共享`x`data的Tensor，且`requires_grad=False`，但是当`x`需要bp的时候，会报告那些in-place的操作。

> However, .data can be unsafe in some cases. Any changes on x.data wouldn’t be tracked by autograd, and the computed gradients would be incorrect if x is needed in a backward pass. A safer alternative is to use x.detach(), which also returns a Tensor that shares data with requires_grad=False, but will have its in-place changes reported by autograd if x is needed in backward.

这里有些绕，可以看下下面的示例代码：

``` py
# 一个简单的计算图：y = sum(x**2)
x = torch.ones((1 ,2))
x.requires_grad_()
y = torch.sum(x**2)
y.backward()
x.grad   # grad: [2, 2, 2]
# 使用.data，在计算完y之后，又改动了x，会造成梯度计算错误
x.grad.zero_()
y = torch.sum(x**2)
data = x.data
data[0, 0] = 2
y.backward()
x.grad   # grad: [4, 2, 2] 错了哦~
# 使用detach，同样的操作，会抛出异常
x.grad.zero_()
y = torch.sum(x**2)
data = x.detach()
data[0, 0] = 2
y.backward()
# 抛出如下异常
# RuntimeError: one of the variables needed for gradient 
# computation has been modified by an inplace operation
```

## 支持0维(scalar)的Tensor

原来的版本中，对Tensor vector（1D Tensor）做索引得到的结果是一个python number，但是对一个Variable vector来说，得到的就是一个`size(1,)`的vector!对于reduction function（如`torch.sum`，`torch.max`）也有这样的问题。

所以我们引入了scalar（0D Tensor）。它可以使用`torch.tensor()` 函数来创建，现在你可以这样做：

``` py
>>> torch.tensor(3.1416)         # 直接创建scalar
tensor(3.1416)
>>> torch.tensor(3.1416).size()  # scalar 是 0D
torch.Size([])
>>> torch.tensor([3]).size()     # 和1D对比
torch.Size([1])
>>>
>>> vector = torch.arange(2, 6)  # 1D的vector
>>> vector
tensor([ 2.,  3.,  4.,  5.])
>>> vector.size()
torch.Size([4])
>>> vector[3]                    # 对1D的vector做indexing，得到的是scalar
tensor(5.)
>>> vector[3].item()             # 使用.item()获取python number
5.0
>>> mysum = torch.tensor([2, 3]).sum()
>>> mysum
tensor(5)
>>> mysum.size()
torch.Size([])
```

### 累积losses
我们在训练的时候，经常有这样的用法：`total_loss += loss.data[0]`。`loss`通常都是由损失函数计算出来的一个标量，也就是包装了`(1,)`大小`Tensor`的`Variable`。在新的版本中，`loss`则变成了0D的scalar。对一个scalar做indexing是没有意义的，应该使用`loss.item()`获取python number。

注意，如果你在做累加的时候没有转换为python number，你的程序可能会出现不必要的内存占用。因为`autograd`会记录调用过程，以便做反向传播。所以，你现在应该写成 `total_loss += loss.item()`。

## 弃用`volatile`标志
`volatile` 标志被弃用了，现在没有任何效果。以前的版本中，一个设置`volatile=True`的`Variable` 表明其不会被`autograd`追踪。现在，被替换成了一个更灵活的上下文管理器，如`torch.no_grad()`，`torch.set_grad_enable(grad_mode)`等。

``` py
>>> x = torch.zeros(1, requires_grad=True)
>>> with torch.no_grad():    # 使用 torch,no_grad()构建不需要track的上下文环境
...     y = x * 2
>>> y.requires_grad
False
>>>
>>> is_train = False
>>> with torch.set_grad_enabled(is_train):   # 在inference的时候，设置不要track
...     y = x * 2
>>> y.requires_grad
False
>>> torch.set_grad_enabled(True)  # 当然也可以不用with构建上下文环境，而单独这样用
>>> y = x * 2
>>> y.requires_grad
True
>>> torch.set_grad_enabled(False)
>>> y = x * 2
>>> y.requires_grad
False
```

## `dtypes`, `devices` 和NumPy风格的构建函数
以前的版本中，我们需要以"tensor type"的形式给出对data type（如`float`或`double`），device type（如cpu或gpu）以及layout（dense或sparse）的限定。例如，`torch.cuda.sparse.DoubleTensor`用来构造一个data type是`double`，在GPU上以及sparse的tensor。

现在我们引入了`torch.dtype`，`torch.device`和`torch.layout`来更好地使用Numpy风格的构建函数。

### `torch.dtype`

下面是可用的 `torch.dtypes` (data types) 和它们对应的tensor types。可以使用`x.dtype`获取。
<table>
   <tr>
      <td>data type</td>
      <td>torch.dtype</td>
      <td>Tensor types</td>
   </tr>
   <tr>
      <td>32-bit floating point</td>
      <td>torch.float32 or torch.float</td>
      <td>torch.*.FloatTensor</td>
   </tr>
   <tr>
      <td>64-bit floating point</td>
      <td>torch.float64 or torch.double</td>
      <td>torch.*.DoubleTensor</td>
   </tr>
   <tr>
      <td>16-bit floating point</td>
      <td>torch.float16 or torch.half</td>
      <td>torch.*.HalfTensor</td>
   </tr>
   <tr>
      <td>8-bit integer (unsigned)</td>
      <td>torch.uint8</td>
      <td>torch.*.ByteTensor</td>
   </tr>
   <tr>
      <td>8-bit integer (signed)</td>
      <td>torch.int8</td>
      <td>torch.*.CharTensor</td>
   </tr>
   <tr>
      <td>16-bit integer (signed)</td>
      <td>torch.int16 or torch.short</td>
      <td>torch.*.ShortTensor</td>
   </tr>
   <tr>
      <td>32-bit integer (signed)</td>
      <td>torch.int32 or torch.int</td>
      <td>torch.*.IntTensor</td>
   </tr>
   <tr>
      <td>64-bit integer (signed)</td>
      <td>torch.int64 or torch.long</td>
      <td>torch.*.LongTensor</td>
   </tr>
</table>

### `torch.device`
`torch.device`包含了device type（如cpu或cuda）和可能的设备id。使用`torch.device('{device_type}')`或`torch.device('{device_type}:{device_ordinal}')`的方式来初始化。 

如果没有指定`device ordinal`，那么默认是当前的device。例如，`torch.device('cuda')`相当于`torch.device('cuda:X')`，其中，`X`是`torch.cuda.current_device()`的返回结果。

使用`x.device`来获取。

### `torch.layout`
`torch.layout`代表了`Tensor`的data layout。 目前支持的是`torch.strided` (dense，也是默认的) 和 `torch.sparse_coo` (COOG格式的稀疏tensor)。

使用`x.layout`来获取。

### 创建`Tensor`（Numpy风格）
你可以使用`dtype`，`device`，`layout`和`requires_grad`更好地控制`Tensor`的创建。

``` py
>>> device = torch.device("cuda:1") 
>>> x = torch.randn(3, 3, dtype=torch.float64, device=device)
tensor([[-0.6344,  0.8562, -1.2758],
        [ 0.8414,  1.7962,  1.0589],
        [-0.1369, -1.0462, -0.4373]], dtype=torch.float64, device='cuda:1')
>>> x.requires_grad  # default is False
False
>>> x = torch.zeros(3, requires_grad=True)
>>> x.requires_grad
True
```

### `torch.tensor(data, ...)`
`torch.tensor`是新加入的`Tesnor`构建函数。它接受一个"array-like"的参数，并将其value copy到一个新的`Tensor`中。可以将它看做`numpy.array`的等价物。不同于`torch.*Tensor`方法，你可以创建0D的Tensor（也就是scalar）。此外，如果`dtype`参数没有给出，它会自动推断。推荐使用这个函数从已有的data，如Python List创建`Tensor`。

``` py
>>> cuda = torch.device("cuda")
>>> torch.tensor([[1], [2], [3]], dtype=torch.half, device=cuda)
tensor([[ 1],
        [ 2],
        [ 3]], device='cuda:0')
>>> torch.tensor(1)               # scalar
tensor(1)
>>> torch.tensor([1, 2.3]).dtype  # type inferece
torch.float32
>>> torch.tensor([1, 2]).dtype    # type inferece
torch.int64
```
我们还加了更多的`Tensor`创建方法。其中有一些`torch.*_like`，`tensor.new_*`这样的形式。

- `torch.*_like`的参数是一个input tensor， 它返回一个相同属性的tensor，除非有特殊指定。
``` py
>>> x = torch.randn(3, dtype=torch.float64)
>>> torch.zeros_like(x)
tensor([ 0.,  0.,  0.], dtype=torch.float64)
>>> torch.zeros_like(x, dtype=torch.int)
tensor([ 0,  0,  0], dtype=torch.int32)
```

- `tensor.new_*`类似，不过它通常需要接受一个指定shape的参数。
``` py
>>> x = torch.randn(3, dtype=torch.float64)
>>> x.new_ones(2)
tensor([ 1.,  1.], dtype=torch.float64)
>>> x.new_ones(4, dtype=torch.int)
tensor([ 1,  1,  1,  1], dtype=torch.int32)
```

为了指定shape参数，你可以使用`tuple`，如`torch.zeros((2, 3))`（Numpy风格）或者可变数量参数`torch.zeros(2, 3)`（以前的版本只支持这种）。

<table>
   <tr>
      <td>Name</td>
      <td>Returned Tensor</td>
      <td>torch.*_likevariant</td>
      <td>tensor.new_*variant</td>
   </tr>
   <tr>
      <td>torch.empty</td>
      <td>unintialized memory</td>
      <td>✔</td>
      <td>✔</td>
   </tr>
   <tr>
      <td>torch.zeros</td>
      <td>all zeros</td>
      <td>✔</td>
      <td>✔</td>
   </tr>
   <tr>
      <td>torch.ones</td>
      <td>all ones</td>
      <td>✔</td>
      <td>✔</td>
   </tr>
   <tr>
      <td>torch.full</td>
      <td>filled with a given value</td>
      <td>✔</td>
      <td>✔</td>
   </tr>
   <tr>
      <td>torch.rand</td>
      <td>i.i.d. continuous Uniform[0, 1)</td>
      <td>✔</td>
      <td></td>
   </tr>
   <tr>
      <td>torch.randn</td>
      <td>i.i.d. Normal(0, 1)</td>
      <td>✔</td>
      <td></td>
   </tr>
   <tr>
      <td>torch.randint</td>
      <td>i.i.d. discrete Uniform in given range</td>
      <td>✔</td>
      <td></td>
   </tr>
   <tr>
      <td>torch.randperm</td>
      <td>random permutation of {0, 1, ..., n - 1}</td>
      <td></td>
      <td></td>
   </tr>
   <tr>
      <td>torch.tensor</td>
      <td>copied from existing data (list, NumPy ndarray, etc.)</td>
      <td></td>
      <td>✔</td>
   </tr>
   <tr>
      <td>torch.from_numpy*</td>
      <td>from NumPy ndarray (sharing storage without copying)</td>
      <td></td>
      <td></td>
   </tr>
   <tr>
      <td>torch.arange, torch.range and torch.linspace</td>
      <td>uniformly spaced values in a given range</td>
      <td></td>
      <td></td>
   </tr>
   <tr>
      <td>torch.logspace</td>
      <td>logarithmically spaced values in a given range</td>
      <td></td>
      <td></td>
   </tr>
   <tr>
      <td>torch.eye</td>
      <td>identity matrix</td>
      <td></td>
      <td></td>
   </tr>
</table>

注：`torch.from_numpy`只接受NumPy `ndarray`作为输入参数。

## 书写设备无关代码（device-agnostic code）
以前版本很难写设备无关代码。我们使用两种方法使其变得简单：

- `Tensor`的`device`属性可以给出其`torch.device`（`get_device`只能获取CUDA tensor）
- 使用`x.to()`方法，可以很容易将`Tensor`或者`Module`在devices间移动（而不用调用`x.cpu()`或者`x.cuda()`。

推荐使用下面的模式：

``` py
# 在脚本开始的地方，指定device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

## 一些代码

# 当你想创建新的Tensor或者Module时候，使用下面的方法
# 如果已经在相应的device上了，将不会发生copy
input = data.to(device)
model = MyModule(...).to(device)
```

## 在`nn.Module`中对于submodule，parameter和buffer名字新的约束
当使用`module.add_module(name, value)`, `module.add_parameter(name, value)` 或者 `module.add_buffer(name, value)`时候不要使用空字符串或者包含`.`的字符串，可能会导致`state_dict`中的数据丢失。如果你在load这样的`state_dict`，注意打补丁，并且应该更新代码，规避这个问题。

## 一个具体的例子
下面是一个code snippet，展示了从0.3.1跨越到0.4.0的不同。

### 0.3.1 version
``` py
  model = MyRNN()
  if use_cuda:
      model = model.cuda()

  # train
  total_loss = 0
  for input, target in train_loader:
      input, target = Variable(input), Variable(target)
      hidden = Variable(torch.zeros(*h_shape))  # init hidden
      if use_cuda:
          input, target, hidden = input.cuda(), target.cuda(), hidden.cuda()
      ...  # get loss and optimize
      total_loss += loss.data[0]

  # evaluate
  for input, target in test_loader:
      input = Variable(input, volatile=True)
      if use_cuda:
          ...
      ...
```

### 0.4.0 version
``` py
  # torch.device object used throughout this script
  device = torch.device("cuda" if use_cuda else "cpu")

  model = MyRNN().to(device)

  # train
  total_loss = 0
  for input, target in train_loader:
      input, target = input.to(device), target.to(device)
      hidden = input.new_zeros(*h_shape)  # has the same device & dtype as `input`
      ...  # get loss and optimize
      total_loss += loss.item()           # get Python number from 1-element Tensor

  # evaluate
  with torch.no_grad():                   # operations inside don't track history
      for input, target in test_loader:
          ...
```

## 附

- [Release Note](https://github.com/pytorch/pytorch/releases/tag/v0.4.0)
- [Documentation](http://pytorch.org/docs/stable/index.html)