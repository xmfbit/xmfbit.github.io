---
title: MacOS Mojave更新之后一定要做这几件事！
date: 2018-10-27 14:57:12
tags:
    - tool
---
很奇怪，对于手机上的APP，我一般能不升级就不升级；但是对于PC上的软件或操作系统更新，则是能升级就升级。。在将手中的MacOS更新到最新版本Mojave后，发现了一些需要手动调节的问题，记录在这里，原谅我标题党的画风。。。
<!-- more -->

## Git等工具
试图使用`git`是出现了如下错误：

```
git clone xx.git
xcrun: error: invalid active developer path (/Library/Developer/CommandLineTools), missing xcrun at: /Library/Developer/CommandLineTools/usr/bin/xcrun
```

解决办法参考[macOS Mojave: invalid active developer path](https://apple.stackexchange.com/questions/254380/macos-mojave-invalid-active-developer-path)中的最高赞回答：
```
xcode-select --install
```

## osxfuse
参考Github讨论帖[osxfuse not compatible with MacOS Mojave](https://github.com/osxfuse/osxfuse/issues/542)，从官网下载最新的3.8.2版本安装即可。

## VSCode等编辑器字体变“瘦”
更新之后，发现VSCode编辑器中的字体变得“很瘦”，不美观。执行下面的命令，并重启机器，应该可以恢复。
```
defaults write -g CGFontRenderingFontSmoothingDisabled -bool NO
```

## Mos Caffine IINA 等APP
Mos可以平滑Mac上外接鼠标的滚动，并调整鼠标滚动方向和Windows相同。更新后发现Mos失灵。这应该是和新版本中更强的权限管理有关，解决办法是在"安全隐私设置" -> “辅助功能”中，先把Mos的勾勾去掉，然后重新勾选。Caffine同样的操作。

IINA是一款Mac上的播放器软件，是我在Mac上的默认播放器。更新后点击媒体文件，发现只是弹出IINA软件的界面，却没有自动播放。解决办法是在媒体文件上右键，在打开方式中重新选择IINA，并勾选默认打开方式选项。

更新新系统后，遇到的坑暂时就这么多。希望能够帮助到需要的人。