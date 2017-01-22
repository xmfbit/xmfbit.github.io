---
title: 使用 Visual Studio 编译 GSL 科学计算库
date: 2016-12-16 19:00:00
tags: 
    - tool
    - gsl
---

GSL是一个GNU支持的科学计算库，提供了很丰富的数值计算方法。[GSL 的项目主页](http://www.gnu.org/software/gsl/)提供的说明来看，它支持如下的科学计算：

（下面的这张表格的HTML使用的是[No-Cruft Excel to HTML Table Converter](http://pressbin.com/tools/excel_to_html_table/index.html)生成的）
{% raw %}
<table>
   <tr>
      <td>Complex Numbers </td>
      <td>Roots of Polynomials</td>
   </tr>
   <tr>
      <td>Special Functions </td>
      <td>Vectors and Matrices</td>
   </tr>
   <tr>
      <td>Permutations </td>
      <td>Sorting</td>
   </tr>
   <tr>
      <td>BLAS Support </td>
      <td>Linear Algebra</td>
   </tr>
   <tr>
      <td>Eigensystems </td>
      <td>Fast Fourier Transforms</td>
   </tr>
   <tr>
      <td>Quadrature </td>
      <td>Random Numbers</td>
   </tr>
   <tr>
      <td>Quasi-Random Sequences </td>
      <td>Random Distributions</td>
   </tr>
   <tr>
      <td>Statistics </td>
      <td>Histograms</td>
   </tr>
   <tr>
      <td>N-Tuples </td>
      <td>Monte Carlo Integration</td>
   </tr>
   <tr>
      <td>Simulated Annealing </td>
      <td>Differential Equations</td>
   </tr>
   <tr>
      <td>Interpolation </td>
      <td>Numerical Differentiation</td>
   </tr>
   <tr>
      <td>Chebyshev Approximation </td>
      <td>Series Acceleration</td>
   </tr>
   <tr>
      <td>Discrete Hankel Transforms </td>
      <td>Root-Finding</td>
   </tr>
   <tr>
      <td>Minimization </td>
      <td>Least-Squares Fitting</td>
   </tr>
   <tr>
      <td>Physical Constants </td>
      <td>IEEE Floating-Point</td>
   </tr>
   <tr>
      <td>Discrete Wavelet Transforms </td>
      <td>Basis splines</td>
   </tr>
</table>
{% endraw %}

GSL的Linux下的配置很简单，照着它的INSTALL文件一步一步来就可以了。CMAKE大法HAO!

``` bash
./configure
make
make install
make clean
```

同样的，GSL也可以在Windows环境下配置，下面记录了如何在Windows环境下使用 Visual Studio 和 CMakeGUI 编译测试GSL。

## 使用CMAKE编译成.SLN文件

打开CMAKEGUI，将输入代码路径选为GSL源代码地址，输出路径设为自己想要的输出路径。点击 “Configure“，选择Visual Studio2013为编译器，点击Finish后会进行必要的配置。然后将表格里面的选项都打上勾，再次点击”Configure“，等待完成之后点击”Generate“。完成之后，就可以在输出路径下看到GSL.sln文件了。

## 使用Visual Studio生成解决方案

使用 Visual Studio 打开刚才生成的.SLN文件，分别在Debug和Release模式下生成解决方案，等待完成即可。

当完成后，你应该可以在路径下看到这样一张图，我们主要关注的文件夹是\bin，\gsl，\Debug和\Release。


## 加入环境变量 

修改环境变量的Path，将\GSL_Build_Path\bin\Debug加入，这主要是为了\Debug文件夹下面的gsl.dll文件。如果不进行这一步的话，一会虽然可以编译，但是却不能运行。

这里顺便注释一句，当使用第三方库的时候，如果需要动态链接库的支持，其中一种方法就是将DLL文件的路径加入到Path中去。

## 建立Visual Studio属性表

Visual Studio可以通过建立工程属性表的方法来配置工程选项，一个OpenCV的例子可以参见Yuanbo She的这篇博文 [Opencv 完美配置攻略 2014 (Win8.1 + Opencv 2.4.8 + VS 2013)](http://my.phirobot.com/blog/2014-02-opencv_configuration_in_vs.html)。

配置文件中主要是包含文件和静态链接库LIB的路径设置。下面把我的贴出来，只需要根据GSL的生成路径做相应修改即可。注意我的属性表中保留了OpenCV的内容，如果不需要的话，尽可以删掉。上面的博文对这张属性表如何配置讲得很清楚，有问题可以去参考。

``` html
<?xml version="1.0" encoding="utf-8"?>
<Project ToolsVersion="4.0" xmlns="http://schemas.microsoft.com/developer/msbuild/2003">
  <ImportGroup Label="PropertySheets" />
  <PropertyGroup Label="UserMacros" />
  <PropertyGroup>
        <IncludePath>$(OPENCV249)\include;E:\GSLCode\gsl-build\;$(IncludePath)</IncludePath>
        <LibraryPath Condition="'$(Platform)'=='Win32'">$(OPENCV249)\x86\vc12\lib;E:\GSLCode\gsl-build\Debug;$(LibraryPath)</LibraryPath>
        <LibraryPath Condition="'$(Platform)'=='X64'">$(OPENCV249)\x64\vc12\lib;E:\GSLCode\gsl-build\Debug;$(LibraryPath)</LibraryPath>
  </PropertyGroup>
  <ItemDefinitionGroup>
        <Link Condition="'$(Configuration)'=='Debug'">
          <AdditionalDependencies>opencv_calib3d249d.lib;opencv_contrib249d.lib;opencv_core249d.lib;opencv_features2d249d.lib;opencv_flann249d.lib;opencv_gpu249d.lib;opencv_highgui249d.lib;opencv_imgproc249d.lib;opencv_legacy249d.lib;opencv_ml249d.lib;opencv_nonfree249d.lib;opencv_objdetect249d.lib;opencv_ocl249d.lib;opencv_photo249d.lib;opencv_stitching249d.lib;opencv_superres249d.lib;opencv_ts249d.lib;opencv_video249d.lib;opencv_videostab249d.lib;gsl.lib;gslcblas.lib;%(AdditionalDependencies)</AdditionalDependencies>
        </Link>
        <Link Condition="'$(Configuration)'=='Release'">
          <AdditionalDependencies>opencv_calib3d249.lib;opencv_contrib249.lib;opencv_core249.lib;opencv_features2d249.lib;opencv_flann249.lib;opencv_gpu249.lib;opencv_highgui249.lib;opencv_imgproc249.lib;opencv_legacy249.lib;opencv_ml249.lib;opencv_nonfree249.lib;opencv_objdetect249.lib;opencv_ocl249.lib;opencv_photo249.lib;opencv_stitching249.lib;opencv_superres249.lib;opencv_ts249.lib;opencv_video249.lib;opencv_videostab249.lib;gsl.lib;gslcblas.lib;%(AdditionalDependencies)</AdditionalDependencies>
        </Link>
  </ItemDefinitionGroup>
  <ItemGroup />
</Project>
```

在以后建立Visual Studio工程的时候，在属性窗口直接添加现有属性表就可以了！

## 测试

在项目网站的教程上直接找到一段代码，进行测试，输出贝塞尔函数的值。


``` cpp
#include <stdio.h>
#include <gsl/gsl_sf_bessel.h>
int main(void)
{
	double x = 5.0;
	double y = gsl_sf_bessel_J0(x);
	printf("J0(%g) = %.18e\n", x, y);
	return 0;
}
```

控制台输出正确：
{% raw %}
<p><img src="http://i.imgur.com/uXhVvwS.jpg" width="600" height="200"></p>
{% endraw %}
