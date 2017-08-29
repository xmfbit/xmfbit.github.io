---
title: 学一点PyQT
date: 2017-08-29 14:07:55
tags:
    - python
    - qt
---
Qt是一个流行的GUI框架，支持C++/Python。这篇文章是我在这两天通过PyQT制作一个串口通信并画图的小程序的时候，阅读[PyQT5的一篇教程](http://zetcode.com/gui/pyqt5/introduction/)时候的记录。
<!-- more -->

## 主要模块
PyQt5中的主要三个模块如下：
- `QtCore`: 和GUI无关的核心功能：文件，时间，多线程等
- `QtGui`：和GUI相关的的东西：事件处理，2D图形，字体和文本等
- `QtWidget`：GUI中的相关组件，例如按钮，窗口等。

其他模块还有`QtBluetooth`，`QtNetwork`等，都是比较专用的模块，用到再说。

## HelloWorld
这里首先给出一段简单的程序，可以在桌面上显示一个窗口。
``` py
import sys
from PyQt5.QtWidgets import QApplication, QWidget

if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = QWidget()
    w.resize(250, 150)
    w.move(300, 300)
    w.setWindowTitle('Simple')
    w.show()

    sys.exit(app.exec_())
```
下面介绍上面代码的含义：
```py
app = QApplication(sys.argv)
```
每个Qt5应用必须首先创建一个application，后面会用到。
``` py
w = QWidget()
w.resize(250, 150)
w.move(300, 300)
w.setWindowTitle('Simple')
w.show()
```
`QtWidget`是所有组件的父类，我们创建了一个`Widget`。没有任何parent widget的Widget会作为窗口出现。接下来，调用其成员函数实现调整大小等功能。最后使用`show()`将其显示出来。

``` py
sys.exit(app.exec_())
```
进入application的主循环，等待事件的触发。当退出程序（也许是通过Ctrl+C实现的）或者关闭窗口（点击关闭）后，主循环退出。

## 添加一个按钮
下面，我们为窗口添加按钮，并为其添加事件响应动作。

参考文档可知，按钮`QPushButton`存在这样的构造函数：
```py
__init__ (self, QWidget parent = None)
```
下面的代码在初始化`QPushButton`实例`btn`时，将`self`作为参数传入，指定了其parent。另外，在指定按钮大小的时候，使用了`sizeHint()`方法自适应调节其大小。

同时，为按钮关联了点击动作。Qt中的事件响应机制通过信号和槽实现。点击事件一旦发生，信号`clicked`会被释放。然后槽相对的处理函数被调用。所谓的槽可以使PyQt提供的slot，或者是任何Python的可调用对象（函数或者实现了`__call__()`方法的对象）。

我们调用了现成的处理函数，来达到关闭窗口的目的。使用`instance()`可以得到当前application实例，调用其`quit()`方法即是退出当前应用，自然窗口就被关闭了。
```py
import sys
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QToolTip
from PyQt5.QtCore import QCoreApplication

class MyWindow(QWidget):
    def __init__(self):
        super(MyWindow, self).__init__()
        self._init_ui()

    def _init_ui(self):
        btn = QPushButton('quit', self)
        btn.clicked.connect(QCoreApplication.instance().quit)
        btn.setToolTip('This is a <b>QPushButton</b> widget')
        btn.move(50, 50)
        btn.resize(btn.sizeHint())

        self.setGeometry(300, 300, 300, 200)
        self.setWindowTitle('Window with Button')
        self.show()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MyWindow()
    sys.exit(app.exec_())
```

## 使用Event处理事件
除了上述的信号和槽的处理方式，也可以使用Event相关的类进行处理。下面的代码在关闭窗口时弹出对话框确认是否关闭。根据用户做出的选择，调用`event.accept()`或`ignore()`完成对事件的处理。

``` py
import sys
from PyQt5.QtWidgets import QWidget, QMessageBox, QApplication

class MyWindow(QWidget):
    def __init__(self):
        super(MyWindow, self).__init__()
        self._init_ui()
    def _init_ui(self):
        self.setGeometry(300, 300, 300, 200)
        self.show()
    def closeEvent(self, ev):
        reply = QMessageBox.question(self, 'Message', 'Are you sure?',
                    QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            ev.accept()
        else:
            ev.ignore()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = MyWindow()
    sys.exit(app.exec_())
```

## 使用Layout组织Widget
组织Widget的方式可以通过绝对位置调整，但是更推荐使用`Layout`组织。

绝对位置是通过指定像素多少来确定widget的大小和位置。这样的话，有以下几个缺点：
- 不同平台可能显示效果不统一；
- 当parent resize的时候，widget大小和位置并不会自动调整
- 编码太麻烦，牵一发而动全身

下面介绍几种常见的`Layout`类。

### Box Layout
有`QVBoxLayout`和`QHBoxLayout`，用来将widget水平或者竖直排列起来。下面的代码通过这两个layout将按钮放置在窗口的右下角。关键的地方在于使用`addSkretch()`方法将一个`QSpacerItem`实例对象插入到了layout中，占据了相应位置。

``` py
import sys
from PyQt5.QtWidgets import (QWidget, QPushButton,
    QHBoxLayout, QVBoxLayout, QApplication)

class MyWindow(QWidget):
    def __init__(self):
        super(MyWindow, self).__init__()
        self._init_ui()

    def _init_ui(self):
        okButton = QPushButton("OK")
        cancelButton = QPushButton("Cancel")

        hbox = QHBoxLayout()
        hbox.addStretch(1)
        hbox.addWidget(okButton)
        hbox.addWidget(cancelButton)

        vbox = QVBoxLayout()
        vbox.addStretch(1)
        vbox.addLayout(hbox)

        self.setLayout(vbox)    

        self.setGeometry(300, 300, 300, 150)
        self.setWindowTitle('Buttons')    
        self.show()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = MyWindow()
    sys.exit(app.exec_())
```

### Grid Layout
`QGridLayout`将空间划分为行列的grid。在向其中添加item的时候，要指定位置。如下，将5行4列的grid设置为计算器的面板模式。
``` py
import sys
from PyQt5.QtWidgets import (QWidget, QGridLayout,
    QPushButton, QApplication)

class MyWindow(QWidget):
    def __init__(self):
        super(MyWindow, self).__init__()
        self._init_ui()

    def _init_ui(self):
        grid = QGridLayout()
        self.setLayout(grid)
        names = ['Cls', 'Bck', '', 'Close',
                 '7', '8', '9', '/',
                '4', '5', '6', '*',
                 '1', '2', '3', '-',
                '0', '.', '=', '+']
        positions = [(i,j) for i in range(5) for j in range(4)]
        for position, name in zip(positions, names):
            if name == '':
                continue
            button = QPushButton(name)
            grid.addWidget(button, *position)

        self.move(300, 150)
        self.setWindowTitle('Calculator')
        self.show()

if __name__ == '__main__':

    app = QApplication(sys.argv)
    win = MyWindow()
    sys.exit(app.exec_())
```

另外，我们还可以通过`setSpacing()`方法设置每个单元格之间的间隔。如果某个widget需要占据多个单元格，可以在`addWidget()`方法中指定要扩展的行列数。

## 事件驱动
PyQt提供了两种事件驱动的处理方式：
- 使用`event`句柄。事件可能是由于UI交互或者定时器等引起，由接收对象进行处理。
- 信号和槽。某个widge交互时，释放相应信号，被槽对应的函数捕获进行处理。

信号和槽可以见上面使用按钮关闭窗口的例子，关键在于调用信号的`connect()`函数将其绑定到某个槽上。Python中的可调用对象都可以作为槽。

而使用event句柄处理时，需要重写override原来的处理函数，见上面使用其在关闭窗口时进行弹窗确认的例子。
