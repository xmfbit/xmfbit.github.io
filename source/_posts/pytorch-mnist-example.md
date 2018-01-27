---
title: toy demo - PyTorch + MNIST
date: 2017-03-04 22:37:44
tags:
     - pytorch
---
本篇文章介绍了使用PyTorch在MNIST数据集上训练MLP和CNN，并记录自己实现过程中的若干问题。
![MNIST](/img/mnist_example.png)
<!-- more -->

## 加载MNIST数据集
PyTorch中提供了MNIST，CIFAR，COCO等常用数据集的加载方法。`MNIST`是`torchvision.datasets`包中的一个类，负责根据传入的参数加载数据集。如果自己之前没有下载过该数据集，可以将`download`参数设置为`True`，会自动下载数据集并解包。如果之前已经下载好了，只需将其路径通过`root`传入即可。

在加载图像后，我们常常需要对图像进行若干预处理。比如减去RGB通道的均值，或者裁剪或翻转图像实现augmentation等，这些操作可以在`torchvision.transforms`包中找到对应的操作。在下面的代码中，通过使用`transforms.Compose()`，我们构造了对数据进行预处理的复合操作序列，`ToTensor`负责将PIL图像转换为Tensor数据（RGB通道从`[0, 255]`范围变为`[0, 1]`）， `Normalize`负责对图像进行规范化。这里需要注意，虽然MNIST中图像都是灰度图像，通道数均为1，但是仍要传入`tuple`。

之后，我们通过`DataLoader`返回一个数据集上的可迭代对象。一会我们通过`for`循环，就可以遍历数据集了。

``` py
import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim

# use cuda or not
use_cuda = torch.cuda.is_available()

trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])

train_set = dset.MNIST(root=root, train=True, transform=trans, download=download)
test_set = dset.MNIST(root=root, train=False, transform=trans)

batch_size = 128

train_loader = torch.utils.data.DataLoader(
                 dataset=train_set,
                 batch_size=batch_size,
                 shuffle=True)
test_loader = torch.utils.data.DataLoader(
                dataset=test_set,
                batch_size=batch_size,
                shuffle=False)

```

## 网络构建
在进行网络构建时，主要通过`torch.nn`包中的已经实现好的卷积层、池化层等进行搭建。例如下面的代码展示了一个具有一个隐含层的MLP网络。`nn.Linear`负责构建全连接层，需要提供输入和输出的通道数，也就是`y = wx+b`中`x`和`y`的维度。

``` py
class MLPNet(nn.Module):
    def __init__(self):
        super(MLPNet, self).__init__()
        self.fc1 = nn.Linear(28*28, 500)
        self.fc2 = nn.Linear(500, 256)
        self.fc3 = nn.Linear(256, 10)
    def forward(self, x):
        x = x.view(-1, 28*28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
```
由于PyTorch可以实现自动求导，所以我们只需实现`forward`过程即可。这里由于池化层和非线性变换都没有参数，所以使用了`nn.functionals`中的对应操作实现。通过看文档，可以发现，一般`nn`里面的各种层，都会在`nn.functionals`里面有其对应。例如卷积层的对应实现，如下所示，需要传入卷积核的权重。

``` py
# With square kernels and equal stride
filters = autograd.Variable(torch.randn(8,4,3,3))
inputs = autograd.Variable(torch.randn(1,4,5,5))
F.conv2d(inputs, filters, padding=1)
```

同样地，我们可以实现LeNet的结构如下。

``` py
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
```

## 训练与测试

在训练时，我们首先应确定优化方法。这里我们使用带动量的`SGD`方法。下面代码中的`optim.SGD`初始化需要接受网络中待优化的`Parameter`列表（或是迭代器），以及学习率`lr`，动量`momentum`。

``` py
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
```

接下来，我们只需要遍历数据集，同时在每次迭代中清空待优化参数的梯度，前向计算，反向传播以及优化器的迭代求解即可。

``` py
## training
model = LeNet()

if use_cuda:
    model = model.cuda()

optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

ceriation = nn.CrossEntropyLoss()

for epoch in xrange(10):
    # trainning
    ave_loss = 0
    for batch_idx, (x, target) in enumerate(train_loader):
        optimizer.zero_grad()
        if use_cuda:
            x, target = x.cuda(), target.cuda()
        x, target = Variable(x), Variable(target)
        out = model(x)
        loss = ceriation(out, target)
        ave_loss = ave_loss * 0.9 + loss.data[0] * 0.1
        loss.backward()
        optimizer.step()
        if (batch_idx+1) % 100 == 0 or (batch_idx+1) == len(train_loader):
            print '==>>> epoch: {}, batch index: {}, train loss: {:.6f}'.format(
                epoch, batch_idx+1, ave_loss)
    # testing
    correct_cnt, ave_loss = 0, 0
    total_cnt = 0
    for batch_idx, (x, target) in enumerate(test_loader):
        if use_cuda:
            x, targe = x.cuda(), target.cuda()
        x, target = Variable(x, volatile=True), Variable(target, volatile=True)
        out = model(x)
        loss = ceriation(out, target)
        _, pred_label = torch.max(out.data, 1)
        total_cnt += x.data.size()[0]
        correct_cnt += (pred_label == target.data).sum()
        # smooth average
        ave_loss = ave_loss * 0.9 + loss.data[0] * 0.1
        
        if(batch_idx+1) % 100 == 0 or (batch_idx+1) == len(test_loader):
            print '==>>> epoch: {}, batch index: {}, test loss: {:.6f}, acc: {:.3f}'.format(
                epoch, batch_idx+1, ave_loss, correct_cnt * 1.0 / total_cnt)

```

当优化完毕后，需要保存模型。这里[官方文档](http://pytorch.org/docs/notes/serialization.html#recommend-saving-models)给出了推荐的方法，如下所示：
``` py
torch.save(model.state_dict(), PATH)   #保存网络参数
the_model = TheModelClass(*args, **kwargs)
the_model.load_state_dict(torch.load(PATH))  #读取网络参数
```

该博客的完整代码可以见：[PyTorch MNIST demo](https://gist.github.com/xmfbit/b27cdbff68870418bdb8cefa86a2d558)。
