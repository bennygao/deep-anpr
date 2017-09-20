# -*- coding:utf-8 -*-
#!/usr/bin/env python
#
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
Routines for training the network.

"""

__all__ = (
    'train',
)

import functools
import glob
import itertools
import multiprocessing
import random
import sys
import time

import cv2
import numpy
import tensorflow as tf

import common
import gen
import model


def code_to_vec(p, code):
    def char_to_vec(c):
        y = numpy.zeros((len(common.CHARS),))
        y[common.CHARS.index(c)] = 1.0
        return y

    c = numpy.vstack([char_to_vec(c) for c in code])

    return numpy.concatenate([[1. if p else 0], c.flatten()])


def read_data(img_glob):
    for fname in sorted(glob.glob(img_glob)):
        im = cv2.imread(fname)[:, :, 0].astype(numpy.float32) / 255.
        code = fname.split("/")[1][9:16]
        p = fname.split("/")[1][17] == '1'
        yield im, code_to_vec(p, code)


def unzip(b):
    xs, ys = zip(*b)
    xs = numpy.array(xs)
    ys = numpy.array(ys)
    return xs, ys


def batch(it, batch_size):
    out = []
    for x in it:
        out.append(x)
        if len(out) == batch_size:
            yield out
            out = []
    if out:
        yield out


def mpgen(f):
    def main(q, args, kwargs):
        try:
            for item in f(*args, **kwargs):
                q.put(item)
        finally:
            q.close()

    @functools.wraps(f)
    def wrapped(*args, **kwargs):
        q = multiprocessing.Queue(3)
        proc = multiprocessing.Process(target=main,
                                       args=(q, args, kwargs))
        proc.start()
        try:
            while True:
                item = q.get()
                yield item
        finally:
            proc.terminate()
            proc.join()

    return wrapped


@mpgen
def read_batches(batch_size):
    g = gen.generate_ims()

    def gen_vecs():
        for im, c, p in itertools.islice(g, batch_size):
            yield im, code_to_vec(p, c)

    while True:
        yield unzip(gen_vecs())


def get_loss(y, y_):
    # Calculate the loss from digits being incorrect.  Don't count loss from
    # digits that are in non-present plates.
    # 以下的注释中，大写字母N表示批次中图片数量
    # y和y_都是shape=(N, 253)的张量，
    # 其中y_[:, 0]表示车牌是否完整出现在128x64的图像中。
    # 253 = (1 + 7 * len(common.CHARS))=(1 + 7 * (10 + 26))
    # 每个车牌上有7个位置，每个位置可能是0~9的数字(10个)或者A~Z的字母(26个)，共计 7 * (10 + 26) = 252。
    # logits和labels都把y和y_向量中第0个元素（表示车牌完整出现在输入图像中概率）去掉，剩下252个元素，
    # 然后再reshape为shape=(N*7, 36)的矩阵。每一行表示车牌上的一个位置，每一列的值是该位置是这个字符的概率。
    with tf.name_scope('softmax_cross_entropy'):
        # logits是把y的维度调整为 (N * 7) x 36 的张量
        logits = tf.reshape(y[:, 1:], [-1, len(common.CHARS)])
        # labels是把y_的维度调整为 (N * 7) x 36 的张量
        labels = tf.reshape(y_[:, 1:], [-1, len(common.CHARS)])

        # 计算softmax之后的交叉熵，softmax_cross_entropy_with_logits返回与logits行数相等的一个一维张量，
        # shape=(N * 7,)
        # 作用：对于logits和labels之间计算softmax交叉熵。就是在离散分类任务的时候度量概率误差的。
        # softmax之后的每一个分量就代表一个类，分量（类）上面的值就是该类的概率。
        # 这个函数并不是计算softmax的函数，只是根据softmax计算分类误差，所以不要吧这个函数当做softmax函数使用。
        # logits和labels必须有相同的形状[batch_size, num_classes]和相同的类型 (either float16, float32, or float64)。
        # 参数：
        # logits: Unscaled log probabilities.
        # labels: 你的labels矩阵，每一行代表一个样本的概率分布（要是你熟悉softmax和onehot encoding的话）
        # dim: 作用的维度，默认是-1，表示最后的那个维度
        # name: 【可选】这个操作的名字
        # 返回: 一个1维的tensor，长度为batch_size,类型和logits一样。
        digits_loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)

        # 把长度等于 (N * 7) 的一维张量reshape为 (N, 7)
        digits_loss = tf.reshape(digits_loss, [-1, 7])
        # 在 shape=(N, 7) 的张量的列维度(axis=1)上做reduce_sum，结果是长度为N的一维张量
        digits_loss = tf.reduce_sum(digits_loss, axis=1)
        # 输入向量 y_[:, 0]不等于0 表示车牌完整的出现在图片中，等于0表示车牌没有完成的出现在图片中。
        # 如果y_[:, 0]不等于0，保留已经计算得到的digits_loss值(*1);
        # 如果y_[:, 0]等于0，则把digits_loss中的值全设为0
        digits_loss *= (y_[:, 0] != 0)
        # 把长度为N的一维张量进行所有维度的reduce，即把所有值加在一起，得到一个值（0维）
        digits_loss = tf.reduce_sum(digits_loss)

    # Calculate the loss from presence indicator being wrong.
    with tf.name_scope('presence_loss'):
        # sigmoid_cross_entropy_with_logits的作用是计算 logits 经 sigmoid 函数激活之后的交叉熵。
        # 对于一个不相互独立的离散分类任务，这个函数作用是去度量概率误差。
        # 比如，在一张图片中，同时包含多个分类目标（大象和狗），那么就可以使用这个函数。
        # 结果返回与logits维度相同的张量。
        presence_loss = tf.nn.sigmoid_cross_entropy_with_logits(
            logits=y[:, :1], labels=y_[:, :1])
        presence_loss = 7 * tf.reduce_sum(presence_loss)

    return digits_loss, presence_loss, digits_loss + presence_loss


def train(learn_rate, report_steps, batch_size, initial_weights=None):
    """
    Train the network.

    The function operates interactively: Progress is reported on stdout, and
    training ceases upon `KeyboardInterrupt` at which point the learned weights
    are saved to `weights.npz`, and also returned.

    :param learn_rate:
        Learning rate to use.

    :param report_steps:
        Every `report_steps` batches a progress report is printed.

    :param batch_size:
        The size of the batches used for training.

    :param initial_weights:
        (Optional.) Weights to initialize the network with.

    :return:
        The learned network weights.

    """
    # 创建前向传播过程model
    # x: 提供前向传播输入的128x64的图像的placeholder
    # y: 前向传播最终结果，是长度为(1 + 7 * len(common.CHARS))的向量。
    # params: 向量，元素为网络中每一层（包括3个卷积层、3个池化层、2个全连接层）的weights和bias。
    with tf.name_scope('input'):
        x, y, params = model.get_training_model()
        # 标准答案向量，维度与结果向量相同。
        y_ = tf.placeholder(tf.float32, [None, 7 * len(common.CHARS) + 1])

    # 创建损失函数
    with tf.name_scope('loss_function'):
        digits_loss, presence_loss, loss = get_loss(y, y_)

    # 创建反向传播过程（优化网络各层的权重参数）
    with tf.name_scope('training_step'):
        # Adam优化，常用的优化方法
        train_step = tf.train.AdamOptimizer(learn_rate).minimize(loss)
        best = tf.argmax(tf.reshape(y[:, 1:], [-1, 7, len(common.CHARS)]), 2)
        correct = tf.argmax(tf.reshape(y_[:, 1:], [-1, 7, len(common.CHARS)]), 2)

    if initial_weights is not None:
        assert len(params) == len(initial_weights)
        assign_ops = [w.assign(v) for w, v in zip(params, initial_weights)]

    init = tf.global_variables_initializer()

    def vec_to_plate(v):
        return "".join(common.CHARS[i] for i in v)

    def do_report():
        r = sess.run([best,
                      correct,
                      tf.greater(y[:, 0], 0),
                      y_[:, 0],
                      digits_loss,
                      presence_loss,
                      loss],
                     feed_dict={x: test_xs, y_: test_ys})
        num_correct = numpy.sum(
            numpy.logical_or(
                numpy.all(r[0] == r[1], axis=1),
                numpy.logical_and(r[2] < 0.5,
                                  r[3] < 0.5)))
        r_short = (r[0][:190], r[1][:190], r[2][:190], r[3][:190])
        for b, c, pb, pc in zip(*r_short):
            print "{} {} <-> {} {}".format(vec_to_plate(c), pc,
                                           vec_to_plate(b), float(pb))
        num_p_correct = numpy.sum(r[2] == r[3])

        print ("B{:3d} {:2.02f}% {:02.02f}% loss: {} "
               "(digits: {}, presence: {}) |{}|").format(
            batch_idx,
            100. * num_correct / (len(r[0])),
            100. * num_p_correct / len(r[2]),
            r[6],
            r[4],
            r[5],
            "".join("X "[numpy.array_equal(b, c) or (not pb and not pc)]
                    for b, c, pb, pc in zip(*r_short)))

    def do_batch():
        if batch_idx % report_steps == 0:
            # 配置运行时需要记录的信息。
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            # 运行时记录运行信息的proto。
            run_metadata = tf.RunMetadata()
            # 训练并获得详细信息
            sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys},
                     options=run_options, run_metadata=run_metadata)
            # 输出日志
            writer.add_run_metadata(run_metadata=run_metadata, tag=("tag%d" % batch_idx), global_step=batch_idx)
            # 向屏幕输出
            do_report()
        else:
            sess.run(train_step,
                     feed_dict={x: batch_xs, y_: batch_ys})

    # 定义GPU参数
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.95)
    # 定义log
    writer = tf.summary.FileWriter("log", tf.get_default_graph())
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        sess.run(init)

        # 如果有上次训练的结果（从命令行参数传入），则用上次训练的结果初始化所有参数。
        if initial_weights is not None:
            sess.run(assign_ops)

        test_xs, test_ys = unzip(list(read_data("test/*.png"))[:50])

        try:
            last_batch_idx = 0
            last_batch_time = time.time()
            batch_iter = enumerate(read_batches(batch_size))
            for batch_idx, (batch_xs, batch_ys) in batch_iter:
                do_batch()
                if batch_idx % report_steps == 0:
                    batch_time = time.time()
                    if last_batch_idx != batch_idx:
                        print "time for 60 batches {}".format(
                            60 * (last_batch_time - batch_time) /
                            (last_batch_idx - batch_idx))
                        last_batch_idx = batch_idx
                        last_batch_time = batch_time

        except KeyboardInterrupt:
            last_weights = [p.eval() for p in params]
            numpy.savez("weights.npz", *last_weights)
            return last_weights

    writer.close()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        f = numpy.load(sys.argv[1])
        initial_weights = [f[n] for n in sorted(f.files,
                                                key=lambda s: int(s[4:]))]
    else:
        initial_weights = None

    train(learn_rate=0.001,
          report_steps=20,
          batch_size=50,
          initial_weights=initial_weights)
