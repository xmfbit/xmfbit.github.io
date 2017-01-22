---
title: CS131_线代基础
date: 2017-01-22 15:38:01
tags:
    - CS131
    - 公开课
---

## 课程简介
CS131课程(Computer Vision: Foundations and Applications)，是斯坦福大学Li Feifei实验室开设的一门计算机视觉入门基础课程，[该课程](http://vision.stanford.edu/teaching/cs131_fall1617/index.html)目的在于为刚接触计算机视觉领域的学生提供基本原理和应用介绍。目前2016年冬季课程刚刚结束。CS131博客系列主要是关于本课的slide知识点总结与作业重点问题归纳，作为个人学习本门课程的心得体会和复习材料。

由于是个人项目，所以会比较随意，只对个人感兴趣的内容做一总结。这篇文章是对课前的[线代基础](http://vision.stanford.edu/teaching/cs131_fall1617/lectures/lecture2_linalg_review_cs131_2016.pdf)做一复习与整理。

## 向量与矩阵
数字图像可以看做二维矩阵，向量是特殊的矩阵，本课程默认的向量都是列向量。
slide中给出了一些矩阵行列式和迹的性质，都比较简单，这里不再多说。
## 矩阵作为线性变换
通过线代知识，我们知道，在线性空间中，如果给定一组基，线性变换可以通过对应的矩阵来进行描述。

### scale变换
对角阵可以用来表示放缩变换。
$$
\begin{bmatrix}
s_x & 0\\\\
0 & s_y
\end{bmatrix}\begin{bmatrix}
x\\\\
y
\end{bmatrix} = \begin{bmatrix}
s_xx\\\\
s_yy
\end{bmatrix}
$$

### 旋转变换
如图所示，逆时针旋转$theta$角度，对应的旋转矩阵为：
![旋转变换](/img/rotation.png)
$$
\mathbf{R} = \begin{bmatrix}
\cos\theta &-\sin\theta \\\\
\sin\theta &\cos\theta
\end{bmatrix}
$$

### 齐次变换(Homogeneous Transform)
只用上面的二维矩阵不能表达平移，使用齐次矩阵可以表达放缩，旋转和平移操作。
$$
\mathbf{H} =\begin{bmatrix}
a & b & t_x\\\\
c & d & t_y\\\\
0 & 0 & 1
\end{bmatrix},\mathbf{H}\begin{bmatrix}
x\\\\
y\\\\
1\\\\
\end{bmatrix}=\begin{bmatrix}
ax+by+t_x\\\\
cx+dy+t_y\\\\
1
\end{bmatrix}
$$
