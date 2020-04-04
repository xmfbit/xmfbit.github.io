layout: post
title: TFRecord 简介
date: 2020-04-03 23:36:29
tags:
     - tf
---

TFRecord是TensorFlow中常用的数据打包格式。通过将训练数据或测试数据打包成TFRecord文件，就可以配合TF中相关的DataLoader / Transformer等API实现数据的加载和处理，便于高效地训练和评估模型。

TF官方tutorial：[TFRecord and tf.Example](https://www.tensorflow.org/tutorials/load_data/tfrecord)

![TFRecord好！](/img/tfrecord_logo.jpeg)

<!-- more -->

# 组成TFReocrd的砖石：`tf.Example`

`tf.Example`是一个Protobuffer定义的message，表达了一组string到bytes value的映射。TFRecord文件里面其实就是存储的序列化的`tf.Example`。如果对Protobuffer不熟悉，可以去看下Google的[文档](https://developers.google.com/protocol-buffers/docs/overview)和[教程](https://developers.google.com/protocol-buffers/docs/pythontutorial)。

## Example 是什么

我们可以具体到相关代码去详细地看下`tf.Example`的构成。作为一个Protobuffer message，它被定义在文件[core/example/example.proto](https://github.com/tensorflow/tensorflow/blob/v2.1.0/tensorflow/core/example/example.proto#L88)中：

``` protobuf
message Example {
  Features features = 1;
};
```

好吧，原来只是包了一层`Features`的message。我们还需要进一步去查找`Features`的message[定义](https://github.com/tensorflow/tensorflow/blob/v2.1.0/tensorflow/core/example/feature.proto#L85-L88)：

``` proto
message Features {
  // Map from feature name to feature.
  map<string, Feature> feature = 1;
};
```

到这里，我们可以看出，`tf.Example`确实表达了一组string到Feature的映射。其中，这个string表示feature name，后面的Feature又是一个message。继续寻找：

``` proto
// Containers for non-sequential data.
message Feature {
  // Each feature can be exactly one kind.
  oneof kind {
    BytesList bytes_list = 1;
    FloatList float_list = 2;
    Int64List int64_list = 3;
  }
};

// 这里摘一个 Int64List 的定义如下，float/bytes同理
message Int64List {
  // 可以看到，如其名所示，表示的是int64数值的列表
  repeated int64 value = 1 [packed = true];
}

```

看起来，是描述了一组各种数据类型的list，包括二进制字节流，float或者int64的数值列表。

## 属于自己的Example

有了上面的分解，要想构造自己数据集的`tf.Example`，就可以一步步组合起来。

首先用下面的几个帮助函数，将给定的Python类型数据转换为对应的Feature。

``` py
# The following functions can be used to convert a value to a type compatible
# with tf.Example.

def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

# 这里我们直接认为value是个标量，如果是tf.Tensor，可以使用
# `tf.io.serialize_tensor`将其序列化为bytes
# `tf.io.parse_tensor`可以反序列化为tf.Tensor

def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
```

有了`Feature`，就可以组成`Features`，只要把对应的名字作为string传进去就行了。

``` py
features_dict = {'image': _bytes_feature(image_data), 'label': _int64_feature(label)}
features = tf.train.Features(feature=features_dict)
```

`Example`自然也就有了：

``` py
example = tf.train.Example(features=features)
```

# TFRecord

TFRecord是一个二进制文件，只能顺序读取。它的数据打包格式如下：

```
uint64 length
uint32 masked_crc32_of_length
byte   data[length]
uint32 masked_crc32_of_data
```

其中，`data[length]`通常是一个`Example`序列化之后的数据。

## 将`Example`写入TFRecord

可以使用python API，将`Example`proto写入TFRecord文件。

``` py
with tf.io.TFRecordWriter(filename) as writer:
    for image_file in image_files: 
        image_data = open(image_file, 'rb').read()
        features = tf.train.Features(feature={'image': _bytes_feature(image_Data)})
        # 得到 example
        example = tf.train.Example(features=features)
        # 通过调用message.SerializeToString() 将其序列化
        writer.write(example.SerializeToString())
```

## 读取TFRecord中的`Example`

通过`tf.data.TFRecordDataset`得到`Dataset`，然后遍历它，并反序列化，就可以得到原始数据。下面的代码段从TFRecord文件中读取刚刚写入的image：

``` py
def parse_from_single_example(example_proto):
    """ 从example message反序列化得到当初写入的内容 """
    # 描述features
    desc = {'image': tf.io.FixedLenFeature([], dtype=tf.string)}
    # 使用tf.io.parse_single_example反序列化
    return tf.io.parse_single_example(example_proto, desc)


def decode_image_from_bytes(image_data):
    """ use cv2.imdecode decode image from raw binary data """
    bytes_array = np.array(bytearray(image_data))
    return cv2.imdecode(bytes_array, cv2.IMREAD_COLOR)


def get_image_from_single_example(example_proto):
    """ get image fom example serialized data """
    data = parse_from_single_example(example_proto)
    image_data = data['image'].numpy()
    # the image_data is str
    # decode the binary bytes to get the image
    return decode_image_from_bytes(image_data)


dataset = tf.data.TFRecordDataset(tfrecord_file)
data_iter = iter(dataset)
first_example = next(data_iter)

first_image = get_image_from_single_example(first_example)
```

或者可以用`map`来将parser的pipeline应用于原dataset：

``` py
# 注意这里不能用get_image_from_single_example
# 因为 `.numpy()` 不能用于静态 Map
image_data = dataset.map(parse_from_single_example)

first_image_data = next(iter(image_data))
image = decode_image_from_bytes(first_image_data['image'].numpy())
```
