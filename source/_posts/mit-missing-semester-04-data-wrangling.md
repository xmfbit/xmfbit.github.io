---
title: MIT Missing Semester - Data Wrangling
date: 2020-03-15 21:47:18
tags:
    - tool
---

这是[MIT Missing Semester系列](https://missing.csail.mit.edu/2020/data-wrangling/)的第四讲。关于vim的第三讲跳过。Data Wrangling在这里的意思是对数据做变换（Transformation）。例如将一个MP4格式的视频转换为AVI，或或者是从日志中提取所需要的结构化文本信息。具体到本课，主要是处理文本信息：如何匹配到我们感兴趣的信息，如果构建一个处理的pipeline等。

## 正则表达式

在很久以前，总结了一篇关于python中的正则表达式的常用用法，竟然也是博客的第一篇文章：[python正则表达式](https://xmfbit.github.io/2014/07/17/python-reg-exp/)。

推荐一个[交互式的正则表达式学习网站](https://regexone.com/)。


A more convenient way is to specify how many repetitions of each character we want using the curly braces notation. For example, a{3} will match the a character exactly three times. Certain regular expression engines will even allow you to specify a range for this repetition such that a{1,3} will match the a character no more than 3 times, but no less than once for example.

This quantifier can be used with any character, or special metacharacters, for example w{3} (three w's), [wxy]{5} (five characters, each of which can be a w, x, or y) and .{2,6} (between two and six of any character).

for example a+ (one or more a's), [abc]+ (one or more of any a, b, or c character) and .* (zero or more of any character)