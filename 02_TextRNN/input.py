# -*- coding:utf-8 -*-
# author: adowu

import tensorflow as tf
import os
import jieba


class Input():
    def __init__(self, params: dict):
        self.data_dir = params.get('data_dir')
        self.sequence_max_length = params.get('sequence_max_length')
        self.batch_size = params.get('batch_size')

    def get_data_dir(self, mode):

        if mode == tf.estimator.ModeKeys.TRAIN:
            file_pattern = '*_train.txt'
            sub = 'train'

        elif mode == tf.estimator.ModeKeys.EVAL:
            file_pattern = '*_eval.txt'
            sub = 'eval'

        else:
            file_pattern = '*_test.txt'
            sub = 'test'

        return os.path.join(self.data_dir, sub, file_pattern)

    def input_fn(self, mode, vocabs: list):

        file_pattern = self.get_data_dir(mode)
        files = tf.gfile.Glob(file_pattern)
        ds = tf.data.TextLineDataset(files)

        def parse(raw):
            """
            在 label 定义的时候有几点需要注意的：
                一、如果label 就是单独的自己的标签，比如 0 或者 1 或者 2
                    那么在计算loss的时候就需要使用：tf.nn.sparse_softmax_cross_entropy_with_logits()
                二、如果对label 进行了one_hot的话，比如[1,0,0],[0,1,0].[0,0,1]
                    那么在计算loss的时候就需要使用：tf.nn.softmax_cross_entropy_with_logits()

                三、另外如果用的是上述第二种方式的时候，由于label是one_hot的了，
                    那么在计算accuracy的时候
                        acc_op = tf.metrics.accuracy(labels=labels, predictions=predictions)
                        predictions就不能是tf.nn.argmax()的结果了这个只有1-D
                        而应该用tf.nn.softmax()的结果作为accuracy metrics的参数来进行计算
                四、另外需要注意的是在dataset里面，需要提前指定输入的维度，而输出label的维度由于只在最后计算
                    loss等的时候才用到，保证和与之计算的logits或者predictions的维度相同就行,当然指定也没事，
                    如果用的不是one_hot label,那么需要这样指定，对于单分类而言
                    tf.reshape(ids, [self.sequence_max_length]),tf.reshape(label, [])
                    如果用的是one_hot label那么需要如下指定，单分类多分类都行
                    tf.reshape(ids, [self.sequence_max_length]),tf.reshape(label, [num_classes])
            """
            raw = raw.decode('utf-8')
            splits = raw.split('\t\t')
            if len(splits) == 2:
                content = splits[0]
                label = int(splits[1])
                # if label == 0:
                #     label = [1,0,0]
                # if label == 1:
                #     label = [0,1,0]
                # if label == 2:
                #     label = [0,0,1]
                ids = segment(content, vocabs, self.sequence_max_length)
            else:
                tf.logging.info("i'm coming")
                ids = [0] * self.sequence_max_length
                label = 0
            if mode == tf.estimator.ModeKeys.PREDICT:
                return [ids]
            return ids, label

        if mode == tf.estimator.ModeKeys.PREDICT:
            ds = ds.repeat(1). \
                shuffle(buffer_size=20 * 1000). \
                map(lambda line: tf.py_func(parse, inp=[line], Tout=[tf.int32])). \
                map(lambda ids: tf.reshape(ids, [self.sequence_max_length])). \
                batch(self.batch_size).prefetch(buffer_size=20 * 1000)

        else:
            ds = ds.repeat(None if mode == tf.estimator.ModeKeys.TRAIN else 1). \
                shuffle(buffer_size=20 * 1000). \
                map(lambda line: tf.py_func(parse, inp=[line], Tout=[tf.int32, tf.int32])). \
                map(lambda ids, label: (
                    tf.reshape(ids, [self.sequence_max_length]),tf.reshape(label, [])
                    )). \
                batch(self.batch_size). \
                prefetch(buffer_size=20 * 1000)
        return ds


def segment(content, vocabs, sequence_max_length):
    ids = list()
    words = jieba.cut(content)
    for word in words:
        if word in vocabs:
            ids.append(vocabs.index(word))
        else:
            ids.append(vocabs.index('UNK'))

    if len(ids) > sequence_max_length:
        ids = ids[:sequence_max_length]
    else:
        ids = ids + [vocabs.index('PAD')]*(sequence_max_length - len(ids))
    return ids


def input_test():
    from run import load_config, enrich_params

    params = dict()
    load_config(params)
    enrich_params(params)
    input = Input(params)
    ds = input.input_fn(mode=tf.estimator.ModeKeys.PREDICT, vocabs=params.get('vocabs'))
    iter = ds.make_one_shot_iterator()
    el = iter.get_next()
    with tf.Session() as sess:
        print(sess.run(el))

if __name__ == '__main__':
    input_test()



