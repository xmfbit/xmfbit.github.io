---
title: 在DigitalOcean上配置Shadowsocks实现IPV4/IPV6翻墙
date: 2017-02-08 17:31:35
tags:
    - tool
---
~~身在天朝，GFW是一个很蛋疼的东西。有人说GFW挡住了天朝与世界其他地方的交流，尤其给科研造成了很多不便；也有人认为GFW挡住了美利坚的“坚船利炮”（Google，Facebook，Tweet等），让中国的互联网企业高速发展。这两种思路都很有道理，然而GFW对计算机相关行业人员造成不便倒也是千真万确，让人在无数次404之后，脱口而出F***。也有人打趣，会不会FQ，已经成为了CS入行的第一道关卡，正好用来刷掉一批人。~~ （2017/10/27）以前这段话真他妈Naive！滑稽。。。


之前我使用GoAgent来FQ，后来又使用了一段时间的免费SS服务，后来机缘巧合从一个印度佬那里挣了一些美元，所以搬到了DigitalOcean上。DO的机器，对我一个学生来说，说实话并不算便宜了，不过好在手里还有一些美元，也在GitHub那里进行了学生认证，算是也可以应付了。

之前我已经在DO的机器上配置过shadowsocks，还顺手给iPad解决了FQ的问题。然而当时没有记录，这次我换了一台机器，机房位于NY，把ss重新配置了一遍，再不做些记录，下次恐怕又要东翻西找。
![佛跳墙](/img/god_use_vpn.png)

<!-- more -->
## 申请机器
在进行GitHub学生认证后，可以使用它发给的一个优惠码注册DO，不过仍然要绑定自己的PayPal账户，先交上五美金。之后，DO会赠送50刀。

申请机器时，直接选择Ubuntu 16.04操作系统，并勾选IPV6 Enable，省去后面的麻烦。

## 安装ss
远程登录后，我们需要安装ss。安装命令很简单。
``` bash
apt-get install python-pip
pip install shadowsocks
```
然而，在安装时，我遇到了一个奇怪的问题，提示我`unsupported locale setting`，后来搜索得知，是语言配置的问题，见[这篇博文](http://www.linfuyan.com/locale_error_unsupported_locale_setting/)，解决办法如下：
``` bash
export LC_ALL=C
```

## 编辑配置文件
之后，进入`/etc`目录，建立一个名叫`shadowsocks.json`的文件（文件名任意，一会对应即可），文件配置内容如下：
```
{
"server":"::",  
"server_port":8388,
"local_address": "127.0.0.1",
"local_port": 1080,
"password":"your_password（任写）",
"timeout":600,
"method":"aes-256-cfb"
}
```
其中第一行写成`::`即是为了IPV6连接。

## 编辑启动项，设置自动启动
之后，我们编辑启动项配置文件，使得ss服务能够开机自动启动。

编辑`/etc/rc.local`文件，在`exit 0`之前添加如下命令。
```
ssserver -c /etc/shadowsocks.json -d start  # 这里的json文件名要相对应
```

之后，使用`reboot`命令重启即可。

## 客户端配置
客户端使用ss，编辑服务器，按照json文件中的内容填写即可。注意密码相对应。

## IOS平台设置
终于把Pad上的翻墙搞定了。。。参考资料为GitHub的相关页面，基本为傻瓜式操作。

- [IPsec VPN 服务器一键安装脚本](https://github.com/hwdsl2/setup-ipsec-vpn/blob/master/README-zh.md)
- [配置 IPsec/L2TP VPN 客户端](https://github.com/hwdsl2/setup-ipsec-vpn/blob/master/docs/clients-zh.md)

首先，使用如下命令在VPN服务器上搭建IPsec服务：

```
wget https://git.io/vpnsetup -O vpnsetup.sh && sudo sh vpnsetup.sh
```

然后按照下面的步骤在IOS平台上进行设置。
![](/img/ipsec_ios_vpn_setting.png)
