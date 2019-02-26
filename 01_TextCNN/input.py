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
            file_pattern = '*_train.csv'
            sub = 'train'

        elif mode == tf.estimator.ModeKeys.EVAL:
            file_pattern = '*_eval.csv'
            sub = 'eval'

        else:
            file_pattern = '*_test.csv'
            sub = 'test'

        return os.path.join(self.data_dir, sub, file_pattern)

    def input_fn(self, mode, vocabs: list):

        file_pattern = self.get_data_dir(mode)
        files = tf.gfile.Glob(file_pattern)
        ds = tf.data.TextLineDataset(files)

        def parse(raw):
            raw = raw.decode('utf-8')
            splits = raw.split()
            if len(splits) == 4:
                content = splits[1] + splits[2]
                label = int(splits[3])
                ids = segment(content, vocabs, self.sequence_max_length)
            else:
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
                tf.reshape(ids, [self.sequence_max_length]),
                tf.reshape(label, [1])
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