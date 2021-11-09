---
title: MIT Missing Semester - Git
date: 2020-07-11 22:24:45
tags:
    - git
    - tool
---

[这一讲](https://missing.csail.mit.edu/2020/version-control/)是关于最流行的版本控制（version control system）工具git的介绍。

![git](/img/mit_missing_semester_06_git_logo.png)

<!-- more -->

## 概述

第一个问题，什么是版本控制系统？版本控制系统常常用来管理源代码（或其他文本格式的文件，二进制文件不推荐）。版本控制系统能够记录文件的历次修改（增删改），并能够记录与之相关的元信息，如修改人，修改原因等。同时也是团队协作的利器。

我们很多人学习git，可能都是采用的自顶向下的学习方法，也就是先接触了git的各种命令。老实说，git的命令行工具入门还是有些门槛的。在这个lecture中，将自底向上地介绍git的基本工作原理。当对git的data model有所了解后，那些命令可能也会变得更加好理解。

## git data model

git的data model是很优雅的。

### snapshot

在git中，一个文件叫做blob（也就是a bunch of bytes），一个文件夹叫做tree。tree下面可以递归地挂tree或者blob。

```
<root> (tree)
|
+- foo (tree)
|  |
|  + bar.txt (blob, contents = "hello world")
|
+- baz.txt (blob, contents = "git is wonderful")
```

最顶上（top-level）那棵树就是一个snapshot。

### 建模历史

用户的更改就是一个个snapshot的更新。snapshot的历史变化就是项目的更新过程。怎么描述这个过程？一种简单的方法是线性表，也就是把历史snapshot按照时间顺序依次排列连接。但这种方法比较单一，git实际采用的是有向无环图DAG。snapshot就是DAG中的每个节点，叫做commit。在下图中，每个节点前面的箭头都指向它的parent。注意，在第三个节点后，历史出现了分叉。在实际中，这可能对应于两个不同的feature开发。git的DAG描述使得这种多个功能并行开发成为可能。

```
o <-- o <-- o <-- o
            ^
             \
              --- o <-- o
```

使用下面的伪代码描述：

```
// a file is a bunch of bytes
type blob = array<byte>

// a directory contains named files and directories
type tree = map<string, tree | blob>

// a commit has parents, metadata, and the top-level tree
type commit = struct {
    parent: array<commit>
    author: string
    message: string
    snapshot: tree
}
```

### object

上面的`blob`，`tree`和`commit`都是`object`。所有的object都有一个[SHA-1 hash](https://en.wikipedia.org/wiki/SHA-1)的字符串来标记。

```
type object = blob | tree | commit
objects = map<string, object>

def store(object):
    id = sha1(object)
    objects[id] = object

def load(id):
    return objects[id]
```

例如，上面给出的那个snapshot，可以使用`git cat-file -p $id`的方式来看到：

```
# git cat-file -p 698281bc680d1995c5f4caaf3359721a5a58d48d
100644 blob 4448adbf7ecd394f42ae135bbeed9676e894af85    baz.txt
040000 tree c68d233a33c5c06e0340e4c224f0afca87c8ce87    foo

# git cat-file -p 4448adbf7ecd394f42ae135bbeed9676e894af85
git is wonderful
```

### refrence

refrence是指向commit的指针，和object的区别是，它是给人读的，而且可以移动。比如master，一般指向主分支的最近的一次commit。

```
# refrence使用一个人类可读好记的string和object的SHA-1 hash串关联
references = map<string, string>

def update_reference(name, id):
    references[name] = id

def read_reference(name):
    return references[name]

def load_reference(name_or_id):
    if name_or_id in references:
        return load(references[name_or_id])
    else:
        return load(name_or_id)
```

除了上述master，git中还使用`HEAD`表示当前所在的snapshot。

### 仓库

git的仓库就是objects和references的集合。使用git命令就是在DAG中更改object，并更新reference。

## Git 常用命令

先介绍一个很好的git教程和参考资料：[pro git](https://git-scm.com/book/zh/v2)

### 入门命令

```
git help <command>: get help for a git command
git init: creates a new git repo, with data stored in the .git directory
git status: tells you what’s going on
git add <filename>: adds files to staging area
git commit: creates a new commit
Write good commit messages!
Even more reasons to write good commit messages!
git log: shows a flattened log of history
git log --all --graph --decorate: visualizes history as a DAG
git diff <filename>: show changes you made relative to the staging area
git diff <revision> <filename>: shows differences in a file between snapshots
git checkout <revision>: updates HEAD and current branch
```

### 分支管理

```
git branch: shows branches
git branch <name>: creates a branch
git checkout -b <name>: creates a branch and switches to it
same as git branch <name>; git checkout <name>
git merge <revision>: merges into current branch
git mergetool: use a fancy tool to help resolve merge conflicts
git rebase: rebase set of patches onto a new base
```

### 和远程仓库交互

```
git remote: list remotes
git remote add <name> <url>: add a remote
git push <remote> <local branch>:<remote branch>: send objects to remote, and update remote reference
git branch --set-upstream-to=<remote>/<remote branch>: set up correspondence between local and remote branch
git fetch: retrieve objects/references from a remote
git pull: same as git fetch; git merge
git clone: download repository from remote
```

### 后悔药

```
git commit --amend: edit a commit’s contents/message
git reset HEAD <file>: unstage a file
git checkout -- <file>: discard changes
```

### 高阶用法

```
git config: Git is highly customizable
git clone --depth=1: shallow clone, without entire version history
git add -p: interactive staging
git rebase -i: interactive rebasing
git blame: show who last edited which line
git stash: temporarily remove modifications to working directory
git bisect: binary search history (e.g. for regressions)
.gitignore: specify intentionally untracked files to ignore
```

## 参考资料

- [pro git](https://git-scm.com/book/zh/v2)，强推，前1-5章必看
- [git for cs scientists](https://eagain.net/articles/git-for-computer-scientists/)，上述git data model的可视化解释
- [oh shit git](https://ohshitgit.com/)，一些git常见误操作的补救方法
- [learning git](https://learngitbranching.js.org/?locale=zh_CN)，一个交互式学习git操作的网站