---
title: Ubuntu Cannot Mount exfat格式硬盘的解决办法
date: 2017-05-04 18:36:29
tags:
    - tool
---
我的移动硬盘为东芝1TB容量，为了能够在Windows和MacOS下使用，我将其格式化为exfat格式。然而我发现这样一来，在Ubuntu14.04下不能挂载。虽然可见盘符，但是却提示`unable to mount`。这篇文章是对解决办法的记录。
![Ubuntu Exfat](/img/ubuntu_exfat.png)

<!-- more -->

## 解决方法
参见[页面](https://askubuntu.com/questions/531919/ubuntu-14-04-cant-mount-exfat-external-hard-disk)，运行以下命令：

```
$ sudo -i  # 获取root权限
# apt-get update
# apt-get install --reinstall exfat-fuse exfat-utils
# mkdir -p /media/user/exfat
# chmod -Rf 777 /media/user/exfat
# fdisk -l
```

之后我发现直接点击盘符的挂载即可，而无需使用他的后续命令。

在弹出驱动器的时候，会出现虽然顺利弹出，但是马上（大概3s），移动硬盘又被读取的情况。所以只能利用间隙，很快地将硬盘取下。不知道会不会有什么损害。所以如果方便的话，还是格式为NTFS格式，再花一些大洋去买Mac上读写NTFS格式硬盘的软件工具吧。。。
