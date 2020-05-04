layout: post
title: 使用 travis 发布博客
date: 2020-03-21 14:51:31
tags:
    - travis
    - tool
---

我的博客之前是通过手动调用`hexo generate`生成`public`，并更新到`master`分支的。最近试了下使用travis直接发布，发现省事了不少。官方的文档其实还是挺全的，不过也碰到了一些坑，记录在这里。

![travis logo](/img/travis_logo.png)
<!-- more -->

## Travis 简介

Travis（/'trævis/）是一款CI（持续集成）工具。[这里](https://www.zhihu.com/question/23444990)有一篇关于持续集成的知乎问答。

> 持续集成强调开发人员提交了新代码之后，立刻进行构建、（单元）测试。根据测试结果，我们可以确定新代码和原有代码能否正确地集成在一起。

在公司内团队内开发某个项目的时候，我们常常也会使用jenkins等工具作为CI的工具。比如当某位同学试图向`master`分支merge代码时，就会触发测试。机器人会在相关MR下评论，通知build和test的结果。

![什么是持续集成](/img/what_is_ci.jpg)

在GitHub的很多项目中，都有CI的身影。如下图所示，在caffe项目的README页面中就显示了该项目目前的CI状态。至于如何在项目中添加这个功能，可以参考[这个页面](https://developer.github.com/v3/repos/statuses/)，这里暂时不多说。

![caffe example](/img/caffe_build_status.png)

而如果我们想在Github自己的项目中使用CI，就可以考虑[Travis](https://travis-ci.org/)。

上面介绍了什么是CI以及travis可以帮助我们进行CI。那为什么可以使用这个功能发布博客吗？因为我们的博客本身是一个依托于Github page功能的静态网站。首先我们有了`username.github.io`这个repo，然后在其`master`分支下放置了hexo生成的静态HTML，就可以看到博客了。想一下之前发布博客的步骤：

- 编写内容
- 使用`hexo generate`生成HTML等（会放在一个`public/`文件夹下）
- 将`public/`发布到repo的master分支下

现在我们就可以把后两步使用travis完成。当我们编写好内容后，将其推送到repo的非master分支，并触发CI的构建，就可以自动完成后两步。

## 配置 Travis

阮一峰的博客里面有一篇[travis教程](http://www.ruanyifeng.com/blog/2017/12/travis_ci_tutorial.html)可以参考。我们这里因为只是一个比较简单的博客发布功能，所以不再展开。

如果你还没有自己的博客，可以去搜索如何使用hexo搭建博客系统，首先确保本地能够跑起来，成功访问自己的博客页面。如果已经有了博客，可以参考[这个repo](https://github.com/hexojs/hexo-starter)，整理下自己的目录结构，尤其是`.gitignore`。注意，这个包含原始网页内容的分支不能是`master`。你可以建立一个叫`hexo`的分支来做这件事。

接下来，在你的账户内添加`Travis CI`：[Travis CI](https://github.com/marketplace/travis-ci)。并在[Applications settings](https://github.com/settings/installations)页面确认Travis可以访问你的repo。这时候应该会重定向到Travis的页面。

打开新窗口，去往[Github token](https://github.com/settings/tokens)生成new token。

在Travis中，找到repo setting，并在`Environment Variables`中，设置name为`GH_TOKEN`，并将上面的token加入。

在你的repo中，checkout到`hexo`分支。并添加`.travis.yml`文件，内容如下：

```
sudo: false
language: node_js
node_js:
  - 10 # use nodejs v10 LTS
cache: npm
branches:
  only:
    - hexo # build hexo branch only
script:
  - hexo generate # generate static files
deploy:
  provider: pages
  skip-cleanup: true
  github-token: $GH_TOKEN
  keep-history: true
  on:
    branch: hexo
  target_branch: master  # 这行很重要
  local-dir: public
```

注意上面一定要设置`target_branch`一项，因为我们需要生成的内容写入`master`分支。关于这些选项的含义，[Travis](https://docs.travis-ci.com/user/deployment/pages/)有相关介绍。不过我是看的[这个](https://bookdown.org/yihui/blogdown/travis-github.html)。这里面的解释更加针对博客部署的场景，建议读一下。

接下来，我们往`hexo`分支上推送内容，就会触发CI并生成网页了！

## 测试

你可以前往[Travis Dashboard](https://travis-ci.com/dashboard)查看自己项目的构建情况。

![travis dashboard](/img/travis_pane.png)

另外，如果你在hexo中也用了`landscape`主题，可能会报fail。解决方法很粗暴，直接删除这个文件就行了：

``` bash
rm themes/landscape/README.md
```

## 注意

我发现travis有两个网站，分别是travis-ci.com和travis-ci.org。这两个网站有什么区别？一句话解释：这两个网站都是可用的travis-ci网站，但目前要用.com那个。具体解释见这个帖子：[What's the difference between travis-ci.org and travis-ci.com?](https://devops.stackexchange.com/questions/1201/whats-the-difference-between-travis-ci-org-and-travis-ci-com)

## 参考资料

- [官方教程](https://hexo.io/docs/github-pages)

上面的官方教程已经说的比较详细了，但是有坑，就是需要设置`target_branch`没有说明。