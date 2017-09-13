# -*- coding:utf-8 -*-
# Copyright (c) 2016 Matthew Earl
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
#     The above copyright notice and this permission notice shall be included
#     in all copies or substantial portions of the Software.
# 
#     THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
#     OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
#     MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN
#     NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
#     DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
#     OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
#     USE OR OTHER DEALINGS IN THE SOFTWARE.


"""
Definition of the neural networks. 

"""

__all__ = (
    'get_training_model',
    'get_detect_model',
    'WINDOW_SHAPE',
)

import tensorflow as tf

import common

WINDOW_SHAPE = (64, 128)


# Utility functions
def weight_variable(shape):
    # tf.truncated_normal(shape, mean, stddev):shape表示生成张量的维度，mean是均值，stddev是标准差。
    # 这个函数产生正太分布，均值和标准差自己设定。这是一个截断的产生正太分布的函数，
    # 就是说产生正太分布的值如果与均值的差值大于两倍的标准差，那就重新生成。
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W, stride=(1, 1), padding='SAME'):
    # tf.nn.conv2d(input, filter, strides, padding, use_cudnn_on_gpu=None, data_format=None, name=None)
    # conv2d实现了卷积层的前向传播算法。
    #
    # @param input 指需要做卷积的输入图像，它要求是一个Tensor，
    # 具有[batch, in_height, in_width, in_channels]这样的shape，
    # 具体含义是[训练时一个batch的图片数量, 图片高度, 图片宽度, 图像通道数]，
    # 注意这是一个4维的Tensor，要求类型为float32和float64其中之一。
    #
    # @param filter 相当于CNN中的卷积核，它要求是一个Tensor，
    # 具有[filter_height, filter_width, in_channels, out_channels]这样的shape，
    # 具体含义是[卷积核的高度，卷积核的宽度，图像通道数，卷积核个数]，要求类型与参数input相同。
    # !有一个地方需要注意，第三维in_channels，就是参数input的第四维
    #
    # @param strides filter的跨度参数，是一个长度为一维向量。
    # 即（image_batch_size_stride、image_height_stride、image_width_stride、image_channels_stride）。
    # 数组第1个元素(输入批次)和最后一个元素(颜色通道)一般都为1。固定要求为1。除非想要在卷积运算中跳过一些数据，从而不将这部分数据予以考虑。
    # 如果希望降低输入的维数，可修改image_height_stride和image_width_stride参数。
    # filter的移动是从上到下、从左到右，所以长指的是行，宽指的是列。
    #
    # @param padding 指定边界填充模式
    # 当卷积核与图像重叠时，它应当落在图像的边界内。有时，两者尺寸可能不匹配，一种较好
    # 的补救策略是对图像缺失的区域进行填充，即边界填充。TensorFlow会用0进行边界填充，
    # 或当卷积核与图像尺寸不匹配，但又不允许卷积核跨越图像边界时，会引发一个错误。
    # tf.nn.conv2d的零填充数量或错误状态是由参数padding控制的，它的取值可以是SAME或VALID。
    # SAME：卷积输出与输入的尺寸相同。这里在计算如何跨越图像时，并不考虑滤波器的尺寸。
    # 选用该设置时，缺失的像素将用0填充，卷积核扫过的像素数将超过图像的实际像素数。
    # VALID：在计算卷积核如何在图像上跨越时，需要考虑滤波器的尺寸。这会使卷积核尽量不越过图像的边界。在某些情形下，可能边界也会被填充。
    # 在计算卷积时，最好能够考虑图像的尺寸，如果边界填充是必要的，则TensorFlow会有一些内置选项。
    # 在大多数比较简单的情形下，SAME都是一个不错的选择。
    #
    # @param use_cudnn_on_gpu bool类型，是否使用cudnn加速，默认为true
    #
    # @returns 返回一个Tensor，这个输出，就是我们常说的feature map，
    # shape仍然是[batch, height, width, channels]这种形式。
    return tf.nn.conv2d(x, W, strides=[1, stride[0], stride[1], 1],
                        padding=padding)


def max_pool(x, ksize=(2, 2), stride=(2, 2)):
    return tf.nn.max_pool(x, ksize=[1, ksize[0], ksize[1], 1],
                          strides=[1, stride[0], stride[1], 1], padding='SAME')


def avg_pool(x, ksize=(2, 2), stride=(2, 2)):
    return tf.nn.avg_pool(x, ksize=[1, ksize[0], ksize[1], 1],
                          strides=[1, stride[0], stride[1], 1], padding='SAME')


def convolutional_layers():
    """
    Get the convolutional layers of the model.

    """
    # 提供training输入的placeholder，3维分别是x轴坐标，y轴坐标和颜色深度
    x = tf.placeholder(tf.float32, [None, None, None])

    # ------------------------------------------------------------------------
    # 第1层
    # 移动窗口尺寸5x5, 原始输入灰度图的颜色深度为1，结果矩阵的深度为48
    W_conv1 = weight_variable([5, 5, 1, 48])

    # 深度48需要48个偏置项
    b_conv1 = bias_variable([48])

    # TODO 卷积层的输入是4维的，第1维是输入批次号，为什么是tf.expand_dims(x, 3)
    # 而不是插入在第0维tf.expand_dims(x, 0)???
    x_expanded = tf.expand_dims(x, 3)

    # 卷积层前向传播，并对结果用激活函数ReLU进行去线性化。
    h_conv1 = tf.nn.relu(conv2d(x_expanded, W_conv1) + b_conv1)

    # 最大池化层，滑动窗口尺寸(2, 2)，步长(2, 2)
    h_pool1 = max_pool(h_conv1, ksize=(2, 2), stride=(2, 2))

    # ------------------------------------------------------------------------
    # Second layer
    W_conv2 = weight_variable([5, 5, 48, 64])
    b_conv2 = bias_variable([64])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool(h_conv2, ksize=(2, 1), stride=(2, 1))

    # Third layer
    W_conv3 = weight_variable([5, 5, 64, 128])
    b_conv3 = bias_variable([128])

    h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
    h_pool3 = max_pool(h_conv3, ksize=(2, 2), stride=(2, 2))

    return x, h_pool3, [W_conv1, b_conv1,
                        W_conv2, b_conv2,
                        W_conv3, b_conv3]


def get_training_model():
    """
    The training model acts on a batch of 128x64 windows, and outputs a (1 +
    7 * len(common.CHARS)) vector, `v`. `v[0]` is the probability that a plate is
    fully within the image and is at the correct scale.
    
    `v[1 + i * len(common.CHARS) + c]` is the probability that the `i`'th
    character is `c`.

    """
    # -------------------------------------------------------------------------
    # 创建卷积层
    # x 提供input的placeholder
    # conv_layer 最后一个池化层，也就是所有卷积层的网络结构。
    # conv_vars 卷积层网络的参数，每个卷积层的weight和bias。
    x, conv_layer, conv_vars = convolutional_layers()

    # -------------------------------------------------------------------------
    # Densely connected layer（创建全连接层）
    # 最后一个池化层：height=8 width=32 deepth=32，全连接层设定节点个数为2048。
    W_fc1 = weight_variable([32 * 8 * 128, 2048])
    b_fc1 = bias_variable([2048])

    # tf.reshape(tensor, shape, name=None)
    # 调整tensor的shape，shape向量中-1表示该维度的大小自动计算，只能出现一次-1。
    conv_layer_flat = tf.reshape(conv_layer, [-1, 32 * 8 * 128])
    # 全连接层的矩阵相乘计算，结果用激活函数ReLU进行去线性化。
    h_fc1 = tf.nn.relu(tf.matmul(conv_layer_flat, W_fc1) + b_fc1)

    # -------------------------------------------------------------------------
    # Output layer（创建输出层）
    # 输出时一个长度为 (1 + 7 * len(common.CHARS)的向量，
    # v[0]表示一个车牌完整地出现在输入的128x64的图片中的概率；
    # v[1 + i * len(common.CHARS) + c]表示第i个字符是'c'的概率。
    W_fc2 = weight_variable([2048, 1 + 7 * len(common.CHARS)])
    b_fc2 = bias_variable([1 + 7 * len(common.CHARS)])

    y = tf.matmul(h_fc1, W_fc2) + b_fc2

    # 返回前向传播的过程
    # x: 输入图像的placeholder
    # y: 最终的结果，长度为(1 + 7 * len(common.CHARS))的向量
    # 向量，元素为每一层（包括所有卷积层、池化层、全连接层）的weights和bias
    return x, y, conv_vars + [W_fc1, b_fc1, W_fc2, b_fc2]


def get_detect_model():
    """
    The same as the training model, except it acts on an arbitrarily sized
    input, and slides the 128x64 window across the image in 8x8 strides.

    The output is of the form `v`, where `v[i, j]` is equivalent to the output
    of the training model, for the window at coordinates `(8 * i, 4 * j)`.

    """
    x, conv_layer, conv_vars = convolutional_layers()

    # Fourth layer
    W_fc1 = weight_variable([8 * 32 * 128, 2048])
    W_conv1 = tf.reshape(W_fc1, [8, 32, 128, 2048])
    b_fc1 = bias_variable([2048])
    h_conv1 = tf.nn.relu(conv2d(conv_layer, W_conv1,
                                stride=(1, 1), padding="VALID") + b_fc1)
    # Fifth layer
    W_fc2 = weight_variable([2048, 1 + 7 * len(common.CHARS)])
    W_conv2 = tf.reshape(W_fc2, [1, 1, 2048, 1 + 7 * len(common.CHARS)])
    b_fc2 = bias_variable([1 + 7 * len(common.CHARS)])
    h_conv2 = conv2d(h_conv1, W_conv2) + b_fc2

    return (x, h_conv2, conv_vars + [W_fc1, b_fc1, W_fc2, b_fc2])
