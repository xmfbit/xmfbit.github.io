---
title: DELL 游匣7559安装Ubuntu和CUDA记录
date: 2017-08-10 11:29:38
tags:
    - ubuntu
---
虽说开源大法好，但是在我的DELL 游匣7559笔记本上安装Ubuntu+Windows双系统可是耗费了我不少精力。这篇博客是我参考[这篇文章](https://hemenkapadia.github.io/blog/2016/11/11/Ubuntu-with-Nvidia-CUDA-Bumblebee.html)成功安装Ubuntu16.04和CUDA的记录。感谢上文作者的记录，我才能够最终解决这个问题。基本流程和上文作者相同，只不过没有安装后续的bumblee等工具，所以本文并不是原创，而更多是翻译和备份。
![开源大法好](/img/install_ubuntu_in_dell_kaiyuandafahao.jpg)
<!-- more -->
## 蛋疼的过往
之前我安装过Ubuntu14.04，但是却不支持笔记本的无线网卡，所以一直很不方便。搜索之后才发现，笔记本使用的无线网卡要到Ubuntu15.10以上才有支持，所以想要安装16.04.结果却发现安装界面都进不去。。。

## 安装Ubuntu
我使用的版本号为Ubuntu16.04.3，使用Windows中的UltraISO制作U盘启动盘。在Windows系统中，通过电池计划关闭快速启动功能，之后重启。在开机出现DELL徽标的时候，按下F12进入BIOS，关闭Security Boot选项。按F10保存并重启，选择U盘启动。

选择“Install Ubuntu”选项，按`e`，找到包含有`quiet splash`的那行脚本，将`quiet splash`替换为以下内容：

``` sh
nomodeset i915.modeset=1 quiet splash
```

之后按F10重启，会进入Ubuntu的安装界面。如何安装Ubuntu这里不再详述。安装完毕之后，重启。出现Ubuntu GRUB引导界面之后，高亮Ubuntu选项（一般来说就是第一个备选项），按`e`，按照上述方法替换`quiet splash`。确定可以进入Ubuntu系统并登陆。

## GRUB设置
下面，修改GRUB设置，避免每次都手动替换。编辑相应配置文件：`sudo vi /etc/default/grub`，找到包含`GRUB_CMDLINE_LINUX_DEFAULT`的那一行，将其修改如下（就是将我们上面每次手动输入的内容直接写到了配置里面）：

``` sh
GRUB_CMDLINE_LINUX_DEFAULT="nomodeset i915.modeset=1 quiet splash"
```

## 更新系统软件
配置更新源（清华的很好用，非教育网也能轻轻松松上700K），使用如下命令更新，

```
sudo apt-get update
sudo apt-get upgrade
sudo apt-get dist-upgrade
sudo apt-get autoremove
```

参考博客中指出，如果这个过程中让你选GRUB文件，要选择保持原有文件。但是我并没有遇到这个问题。可能是由于我的Ubuntu版本已经是16.04中目前最新的了？

由于后续有较多的终端文件编辑操作，建议这时候顺便安装Vim。
``` sh
sudo apt-get install vim
```

更新完之后，重启，确认可以正常登陆系统。

## 移除原有的Nvidia和Nouveau驱动
按下ALT+CTL+F1，进入虚拟终端，首先关闭lightdm服务。这项操作之后会比较经常用到。
``` sh
sudo service lightdm stop
```

之后，执行卸载操作：
``` sh
sudo apt-get remove --purge nvidia*
sudo apt-get remove --purge bumblebee*
sudo apt-get --purge remove xserver-xorg-video-nouveau*
```

编辑配置文件，`/etc/modprobe.d/blacklist.conf`，将Nouveau加入到黑名单中：
```
blacklist nouveau
blacklist lbm-nouveau
alias nouveau off
alias lbm-nouveau off
options nouveau modeset=0
```

编辑`/etc/init/gpu-manager.conf`文件，将其前面几行注释掉，改成下面的样子，停止gpu-manager服务：
``` sh
# Comment these start on settings ; GPU Manager ruins our work
#start on (starting lightdm
#          or starting kdm
#          or starting xdm
#          or starting lxdm)
task
exec gpu-manager --log /var/log/gpu-manager.log
```

之后，更新initramfs并重启。
``` sh
sudo update-initramfs -u -k all
```

重启后，确定可以正常登陆系统。并使用下面的命令确定Nouveau被卸载掉了：
``` sh
# 正常情况下，下面的命令应该不产生任何输出
lsmod | grep nouveau
```

并确定关闭了gpu-manager服务：
``` sh
sudo service gpu-manager stop
```

至此，Ubuntu系统算是安装完毕了。如果没有使用CUDA的需求，可以从这里开始，安安静静地做一个使用Ubuntu的美男子/小仙女了。
![微笑中带着疲惫](/img/install_ubuntu_in_dell_weixiaodaizhepibei.jpg)

## 安装CUDA
鉴于国内坑爹的连接资本主义世界的网络环境，建议还是先去Nvidia的官网把CUDA离线安装包下载下来再安装。我使用的是CUDA-8.0-linux.deb安装包。

按ALT+CTL+F1进入虚拟终端，停止lightdm服务，并安装一些可能要用到的包。
``` sh
sudo service lightdm stop
sudo apt-get install linux-headers-$(uname -r)
sudo apt-get install mesa-utils
```

安装CUDA包：
``` sh
sudo dpkg -i YOUR_CUDA_DEB_PATH
sudo apt-get update
sudo apt-get install cuda-8-0
sudo apt-get autoremove
```
安装完毕之后使用`sudo reboot`重启，确定能够正常登陆系统。

在这个过程中，作者提到登录界面会出现两次，再次重启之后没有这个问题了。我也遇到了相同的情况。所以，不要慌！

## 测试CUDA
我们来测试一下CUDA。首先，依照你使用shell的不同，将环境变量加入到`~/.bashrc`或者`~/.zshrc`（如果使用zsh）中去。
``` sh
export PATH="$PATH:/usr/local/cuda-8.0/bin"
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda-8.0/lib"
```

接下来，我们将使用CUDA自带的example进行测试：
``` sh
# 导入我们刚加入的环境变量
source ~/.bashrc
cd /usr/local/cuda-8.0/bin
# 将CUDA example拷贝到$HOME下
./cuda-install-samples-8.0.sh ~
# 进入拷贝到的那个目录 build
cd ~/NVIDIA_CUDA-8.0_Samples
make -j12
# 自己挑选几个目录进去运行编译生成的可执行文件测试吧~
```

## Last But Not Least
安装完CUDA之后，不要随便更新系统！！！否则可能会损坏你的Kernel和Xserver。
![微笑就好](/img/install_ubuntu_in_dell_weixiaojiuhao.jpg)
