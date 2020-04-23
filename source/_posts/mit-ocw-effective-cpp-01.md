layout: post
title: MIT OCW - Effective Pragramming in C/C++ 02
date: 2020-04-23 23:16:04
tags:
    - 公开课
    - ocw
    - 6.S096
---

跳过了Lecture 01，将Lecture 02作为MIT OCW Effective C++编程课的正式第一课，记录一些要点。

# 内存模型

栈上的内存和堆上的内存。

## 栈内存

下面展示了一个函数调用栈的增长。

![函数调用栈](/img/mit_ocw_effe_cpp_02_stack_mem.png)

## 堆内存

使用`malloc`动态alloc在堆上。记得要`free`。

## 语法糖：array indexing

`T array[]`和`T *array = malloc(...)`都是指向连续内存的指针。下面是完全一样的。array indexing只是一个语法糖。

``` cpp
int array[10];

array[0]

*(array + 0)
```

# 浮点数

IEEE 754浮点数规范，符号位，阶数和尾数。

![浮点数表示](/img/mit_ocw_effe_cpp_02_float_in_mem.png)

# 作业

打印浮点数的二进制表示：

``` cpp
#include <stdio.h>

union float_bits {
    float f;
    unsigned int bits;
};

void print_hex(float f) {
    union float_bits t;
    t.f = f;
    printf("hex: 0x%x\n", t.bits);
}

// print binary
void print_binary(float f) {
    union float_bits t;
    t.f = f;
    if (t.bits >> 31) {
        printf("-");
    }
    printf("1.");
    // get mantissa, which is the last 23 bits
    for (int i = 0; i < 23; ++i) {
        printf("%d", (t.bits >> (22 - i)) & 0x1);
    }
    printf(" * 2^");
    int e = (t.bits & 0x7f800000) >> 23;
    printf("%d\n", e - 0x7f);
}

int main() {
    float f;
    scanf("%f", &f);
    print_hex(f);
    print_binary(f);
    return 0;
}
```

和标准答案相比，没有考虑特殊情况，而且代码写的比较乱。位运算很长时间没用过，很生疏。

``` cpp
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#define ABSOLUTE_WIDTH 31
#define MANTISSA_WIDTH 23
#define EXPONENT_WIDTH 8
#define EXPONENT_MASK 0xffu
#define MANTISSA_MASK 0x007fffffu
#define EXPONENT_BIAS 127
union float_bits {
  float f;
  uint32_t bits;
};
void print_float(FILE *output, float f) {
  union float_bits t;
  t.f = f;
  uint32_t sign_bit = (t.bits >> ABSOLUTE_WIDTH);
  uint32_t exponent = (t.bits >> MANTISSA_WIDTH) & EXPONENT_MASK;
  uint32_t mantissa = (t.bits & MANTISSA_MASK);
  if (sign_bit != 0) {
    fprintf(output, "-");
  }
  if (exponent > 2 * EXPONENT_BIAS) {
    fprintf(output, "Inf\n"); /* Infinity */
    return;
  } else if (exponent == 0) {
    fprintf(output, "0."); /* Zero or Denormal */
    exponent = (mantissa != 0) ? exponent + 1 : exponent;
  } else {
    fprintf(output, "1."); /* Usual */
  }
  for (int k = MANTISSA_WIDTH - 1; k >= 0; --k) {
    fprintf(output, "%d", (mantissa >> k) & 1);
  }
  if (exponent != 0 || mantissa != 0) {
    fprintf(output, " * 2^%d\n", (int)(exponent - EXPONENT_BIAS));
  }
}
int main() {
  FILE *input = fopen("floating.in", "r"), *output = fopen("floating.out", "w");
  size_t N;
  float f;
  fscanf(input, "%zu", &N);
  for (size_t i = 0; i < N; ++i) {
    fscanf(input, "%f", &f);
    print_float(output, f);
  }
  fclose(input);
  fclose(output);
  return 0;
}
```