---
title: PyTorch简介
date: 2017-02-25 19:23:39
tags:
    - pytorch
---
这是一份阅读PyTorch教程的笔记，记录jupyter notebook的关键点。原地址位于[GitHub repo](https://github.com/pytorch/tutorials/blob/master/Deep%20Learning%20with%20PyTorch.ipynb)。
![PyTorch Logo](/img/pytorch_logo.png)

<!-- more -->
## PyTorch简介
[PyTorch](https://github.com/pytorch/pytorch)是一个较新的深度学习框架。从名字可以看出，其和Torch不同之处在于PyTorch使用了Python作为开发语言，所谓“Python first”。一方面，使用者可以将其作为加入了GPU支持的`numpy`，另一方面，PyTorch也是强大的深度学习框架。

目前有很多深度学习框架，PyTorch主推的功能是动态网络模型。例如在Caffe中，使用者通过编写网络结构的`prototxt`进行定义，网络结构是不能变化的。而PyTorch中的网络结构不再只能是一成不变的。同时PyTorch实现了多达上百种op的自动求导（AutoGrad）。

## Tensors
`Tensor`，即`numpy`中的多维数组。上面已经提到过，PyTorch对其加入了GPU支持。同时，PyTorch中的`Tensor`可以与`numpy`中的`array`很方便地进行互相转换。

通过`Tensor(shape)`便可以创建所需要大小的`tensor`。如下所示。

``` py
x = torch.Tensor(5, 3)  # construct a 5x3 matrix, uninitialized
# 或者随机填充
y = torch.rand(5, 3)    # construct a randomly initialized matrix
# 使用size方法可以获得tensor的shape信息，torch.Size 可以看做 tuple
x.size()                # out: torch.Size([5, 3])
```

PyTorch中已经实现了很多常用的`op`，如下所示。

``` py
# addition: syntax 1
x + y                  # out: [torch.FloatTensor of size 5x3]

# addition: syntax 2
torch.add(x, y)        # 或者使用torch包中的显式的op名称

# addition: giving an output tensor
result = torch.Tensor(5, 3)  # 预先定义size
torch.add(x, y, out=result)  # 结果被填充到变量result

# 对于加法运算，其实没必要这么复杂
out = x + y                  # 无需预先定义size

# torch包中带有下划线的op说明是就地进行的，如下所示
# addition: in-place
y.add_(x)              # 将x加到y上
# 其他的例子: x.copy_(y), x.t_().
```

PyTorch中的元素索引方式和`numpy`相同。

``` py
# standard numpy-like indexing with all bells and whistles
x[:,1]                 # out: [torch.FloatTensor of size 5]
```

对于更多的`op`，可以参见PyTorch的[文档页面](http://pytorch.org/docs/torch.html)。

`Tensor`可以和`numpy`中的数组进行很方便地转换。并且转换前后并没有发生内存的复制（这里文档里面没有明说？），所以修改其中某一方的值，也会引起另一方的改变。如下所示。

``` py
# Tensor 转为 np.array
a = torch.ones(5)    # out: [torch.FloatTensor of size 5]
# 使用 numpy方法即可实现转换
b = a.numpy()        # out: array([ 1.,  1.,  1.,  1.,  1.], dtype=float32)
# 注意！a的值的变化同样引起b的变化
a.add_(1)
print(a)
print(b)             # a b的值都变成2

# np.array 转为Tensor
import numpy as np
a = np.ones(5)
# 使用torch.from_numpy即可实现转换
b = torch.from_numpy(a)  # out: [torch.DoubleTensor of size 5]
np.add(a, 1, out=a)
print(a)
print(b)            # a b的值都变为2
```

PyTorch中使用GPU计算很简单，通过调用`.cuda()`方法，很容易实现GPU支持。

``` py
# let us run this cell only if CUDA is available
if torch.cuda.is_available():
    print('cuda is avaliable')
    x = x.cuda()
    y = y.cuda()
    x + y          # 在GPU上进行计算
```

## Neural Network
说完了数据类型`Tensor`，下一步便是如何实现一个神经网络。首先，对[自动求导](http://pytorch.org/docs/autograd.html)做一说明。

我们需要关注的是`autograd.Variable`。这个东西包装了`Tensor`。一旦你完成了计算，就可以使用`.backward()`方法自动得到（以该`Variable`为叶子节点的那个）网络中参数的梯度。`Variable`有一个名叫`data`的字段，可以通过它获得被包装起来的那个原始的`Tensor`数据。同时，使用`grad`字段，可以获取梯度（也是一个`Variable`）。

`Variable`是计算图的节点，同时`Function`实现了变量之间的变换。它们互相联系，构成了用于计算的无环图。每个`Variable`有一个`creator`的字段，表明了它是由哪个`Function`创建的（除了用户自己显式创建的那些，这时候`creator`是`None`）。

当进行反向传播计算梯度时，如果`Variable`是标量（比如最终的`loss`是欧氏距离或者交叉熵），那么`backward()`函数不需要参数。然而如果`Variable`有不止一个元素的时候，需要为其中的每个元素指明其（由上层传导来的）梯度（也就是一个和`Variable`shape匹配的`Tensor`）。看下面的说明代码。

``` py
from torch.autograd import Variable
x = Variable(torch.ones(2, 2), requires_grad = True)
x     # x 包装了一个2x2的Tensor
"""
Variable containing:
 1  1
 1  1
[torch.FloatTensor of size 2x2]
"""
# Variable进行计算
# y was created as a result of an operation,
# so it has a creator
y = x + 2
y.creator    # out: <torch.autograd._functions.basic_ops.AddConstant at 0x7fa1cc158c08>

z = y * y * 3  
out = z.mean()   # out: Variable containing: 27 [torch.FloatTensor of size 1]

# let's backprop now
out.backward()  # 其实相当于 out.backward(torch.Tensor([1.0]))

# print gradients d(out)/dx
x.grad
"""
Variable containing:
 4.5000  4.5000
 4.5000  4.5000
[torch.FloatTensor of size 2x2]
"""
```

下面的代码就是结果不是标量，而是普通的`Tensor`的例子。
``` py
# 也可以通过Tensor显式地创建Variable
x = torch.randn(3)
x = Variable(x, requires_grad = True)
# 一个更复杂的 op例子
y = x * 2
while y.data.norm() < 1000:
    y = y * 2

# 计算 dy/dx
gradients = torch.FloatTensor([0.1, 1.0, 0.0001])
y.backward(gradients)
x.grad
"""
Variable containing:
  204.8000
 2048.0000
    0.2048
[torch.FloatTensor of size 3]
"""
```

说完了NN的构成元素`Variable`，下面可以介绍如何使用PyTorch构建网络了。这部分主要使用了`torch.nn`包。我们自定义的网络结构是由若干的`layer`组成的，我们将其设置为 `nn.Module`的子类，只要使用方法`forward(input)`就可以返回网络的`output`。下面的代码展示了如何建立一个包含有`conv`和`max-pooling`和`fc`层的简单CNN网络。

``` py
import torch.nn as nn                 # 以我的理解，貌似有参数的都在nn里面
import torch.nn.functional as F       # 没有参数的（如pooling和relu）都在functional里面？

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5) # 1 input image channel, 6 output channels, 5x5 square convolution kernel
        self.conv2 = nn.Conv2d(6, 16, 5)
        # 一会可以看到，input是1x32x32大小的。经过计算，conv-pooling-conv-pooling后大小为16x5x5。
        # 所以fc层的第一个参数是 16x5x5
        self.fc1   = nn.Linear(16*5*5, 120) # an affine operation: y = Wx + b
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 10)

    def forward(self, x):
        # 你可以发现，构建计算图的过程是在前向计算中完成的，也许这可以让你体会到所谓的动态图结构
        # 同时，我们无需实现 backward，这是被自动求导实现的
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2)) # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv2(x)), 2) # If the size is a square you can only specify a single number
        x = x.view(-1, self.num_flat_features(x))  # 把它拉直
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:] # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

# 实例化Net对象
net = Net()
net     # 给出了网络结构
"""
Net (
  (conv1): Conv2d(1, 6, kernel_size=(5, 5), stride=(1, 1))
  (conv2): Conv2d(6, 16, kernel_size=(5, 5), stride=(1, 1))
  (fc1): Linear (400 -> 120)
  (fc2): Linear (120 -> 84)
  (fc3): Linear (84 -> 10)
)
"""
```

我们可以列出网络中的所有参数。

``` py
params = list(net.parameters())
print(len(params))      # out: 10, 5个权重，5个bias
print(params[0].size())  # conv1's weight out: torch.Size([6, 1, 5, 5])
print(params[1].size())  # conv1's bias, out: torch.Size([6])
```

给出网络的输入，得到网络的输出。并进行反向传播梯度。

``` py
input = Variable(torch.randn(1, 1, 32, 32))
out = net(input)         # 重载了()运算符？
net.zero_grad()          # bp前，把所有参数的grad buffer清零
out.backward(torch.randn(1, 10))
```

注意一点，`torch.nn`只支持mini-batch。所以如果你的输入只有一个样例的时候，使用`input.unsqueeze(0)`人为给它加上一个维度，让它变成一个4-D的`Tensor`。

## 网络训练
给定target和网络的output，就可以计算loss函数了。在`torch.nn`中已经[实现好了一些loss函数](http://pytorch.org/docs/nn.html#loss-functions)。

``` py
output = net(input)
target = Variable(torch.range(1, 10))  # a dummy target, for example
# 使用平均平方误差，即欧几里得距离
criterion = nn.MSELoss()
loss = criterion(output, target)
loss
"""
Variable containing:
 38.6049
[torch.FloatTensor of size 1]
"""
```

网络的整体结构如下所示。

```
input -> conv2d -> relu -> maxpool2d -> conv2d -> relu -> maxpool2d  
      -> view -> linear -> relu -> linear -> relu -> linear
      -> MSELoss
      -> loss
```

我们可以使用`previous_functions`来获得该节点前面`Function`的信息。

```
# For illustration, let us follow a few steps backward
print(loss.creator) # MSELoss
print(loss.creator.previous_functions[0][0]) # Linear
print(loss.creator.previous_functions[0][0].previous_functions[0][0]) # ReLU
"""
<torch.nn._functions.thnn.auto.MSELoss object at 0x7fa18011db40>
<torch.nn._functions.linear.Linear object at 0x7fa18011da78>
<torch.nn._functions.thnn.auto.Threshold object at 0x7fa18011d9b0>
"""
```

进行反向传播后，让我们查看一下参数的变化。

``` py
# now we shall call loss.backward(), and have a look at conv1's bias gradients before and after the backward.
net.zero_grad() # zeroes the gradient buffers of all parameters
print('conv1.bias.grad before backward')
print(net.conv1.bias.grad)
loss.backward()
print('conv1.bias.grad after backward')
print(net.conv1.bias.grad)
```

计算梯度后，自然需要更新参数了。简单的方法可以自己手写：

``` py
learning_rate = 0.01
for f in net.parameters():
    f.data.sub_(f.grad.data * learning_rate)
```

不过，`torch.optim`中已经提供了若干优化方法（SGD, Nesterov-SGD, Adam, RMSProp, etc）。如下所示。

``` py
import torch.optim as optim
# create your optimizer
optimizer = optim.SGD(net.parameters(), lr = 0.01)
# in your training loop:
optimizer.zero_grad() # zero the gradient buffers
output = net(input)
loss = criterion(output, target)
loss.backward()
optimizer.step() # Does the update
```

## 数据载入
由于PyTorch的Python接口和`np.array`之间的方便转换，所以可以使用其他任何数据读入的方法（例如OpenCV等）。特别地，对于vision的数据，PyTorch提供了`torchvision`包，可以方便地载入常用的数据集（Imagenet, CIFAR10, MNIST, etc），同时提供了图像的各种变换方法。下面以CIFAR为例子。

``` py
import torchvision
import torchvision.transforms as transforms

# The output of torchvision datasets are PILImage images of range [0, 1].
# We transform them to Tensors of normalized range [-1, 1]
# Compose: Composes several transforms together.
# see http://pytorch.org/docs/torchvision/transforms.html?highlight=transforms
transform=transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                             ])   # torchvision.transforms.Normalize(mean, std)
# 读取CIFAR10数据集                             
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
# 使用DataLoader
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)
# Test集，设置train = False
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                          shuffle=False, num_workers=2)
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
```

接下来，我们对上面部分的CNN网络进行小修，设置第一个`conv`层接受3通道的输入。并使用交叉熵定义loss。

``` py
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool  = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1   = nn.Linear(16*5*5, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16*5*5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net()
# use a Classification Cross-Entropy loss
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
```

接下来，我们进行模型的训练。我们loop over整个dataset两次，对每个mini-batch进行参数的更新。并且设置每隔2000个mini-batch打印一次loss。

``` py
for epoch in range(2): # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs
        inputs, labels = data

        # wrap them in Variable
        inputs, labels = Variable(inputs), Variable(labels)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()        
        optimizer.step()

        # print statistics
        running_loss += loss.data[0]
        if i % 2000 == 1999: # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' % (epoch+1, i+1, running_loss / 2000))
            running_loss = 0.0
print('Finished Training')
```

我们在测试集上选取一个mini-batch（也就是4张，见上面`testloader`的定义），进行测试。

``` py
dataiter = iter(testloader)
images, labels = dataiter.next()   # 得到image和对应的label
outputs = net(Variable(images))

# the outputs are energies for the 10 classes.
# Higher the energy for a class, the more the network
# thinks that the image is of the particular class
# So, let's get the index of the highest energy
_, predicted = torch.max(outputs.data, 1)   # 找出分数最高的对应的channel，即为top-1类别

print('Predicted: ', ' '.join('%5s'% classes[predicted[j][0]] for j in range(4)))
```

测试一下整个测试集合上的表现。

``` py
correct = 0
total = 0
for data in testloader:     # 每一个test mini-batch
    images, labels = data
    outputs = net(Variable(images))
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum()

print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))
```

对哪一类的预测精度更高呢？

``` py
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
for data in testloader:
    images, labels = data
    outputs = net(Variable(images))
    _, predicted = torch.max(outputs.data, 1)
    c = (predicted == labels).squeeze()
    for i in range(4):
        label = labels[i]
        class_correct[label] += c[i]
        class_total[label] += 1
```

上面这些训练和测试都是在CPU上进行的，如何迁移到GPU？很简单，同样用`.cuda()`方法就行了。

``` py
net.cuda()
```

不过记得在每次训练测试的迭代中，`images`和`label`也要传送到GPU上才可以。

``` py
inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
```
## 更多的例子和教程
[更多的例子](https://github.com/pytorch/examples)
[更多的教程](https://github.com/pytorch/tutorials)
