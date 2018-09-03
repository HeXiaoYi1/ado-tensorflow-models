# -*- coding:utf-8 -*-
"""
@contact: adonis_wu@outlook.com
@file: tf_txt2batch.py
@time: 2018/9/3 16:01
"""
__author__ = '🍊 Adonis Wu 🍊'

import tensorflow as tf
from tensorflow.contrib.layers.python.ops.sparse_ops import dense_to_sparse_tensor
from tensorflow.python.lib.io import file_io
import json
import os


def csv_input_fn(mode: tf.estimator.ModeKeys, data_dir: str):
    def parse(raw):
        if mode == tf.estimator.ModeKeys.PREDICT:
            columns = tf.decode_csv(raw, record_defaults=[[0], ['<UNK>'], ['<UNK>']])

            df = dict(zip(['id', 'article', 'word_seg'], columns))
        else:
            columns = tf.decode_csv(raw, record_defaults=[[0], ['<UNK>'], ['<UNK>'], ['0']])

            df = dict(zip(['id', 'article', 'word_seg', 'class'], columns))

        #   1-d outsize, dense tensor ,if use batch() api in dataset, you should dense_to_sparse_tensor
        feature = tf.string_split(tf.reshape(df.pop('word_seg'), [1])).values

        feature = dense_to_sparse_tensor(feature)

        features = {
            'feature': feature
        }
        if mode == tf.estimator.ModeKeys.PREDICT:
            return features

        target = df.pop('class')

        labels = {
            'label': target
        }

        return features, labels

    file_paths = file_io.get_matching_files(data_dir)
    ds = tf.data.TextLineDataset(file_paths, buffer_size=10 * 1024 * 1024)
    if mode == tf.estimator.ModeKeys.TRAIN:
        """
            data_set = ds.repeat(None).shuffle(buffer_size=1024 * 1024).map(parse_csv_row).padded_batch(
            params.batch_size, padded_shapes=self.padded_shapes)
            padded_batch need each sentence is not very big
        """
        data_set = ds.repeat(None).shuffle(buffer_size=1024 * 1024).map(parse).batch(16).prefetch(buffer_size=None)
    elif mode == tf.estimator.ModeKeys.EVAL:
        data_set = ds.repeat(None).take(30000).map(parse).batch(16).prefetch(buffer_size=None)

    else:
        data_set = ds.repeat(None).skip(1).take(30000).map(parse).batch(16).prefetch(buffer_size=None)

    return data_set


def json_input_fn(mode: tf.estimator.ModeKeys, data_dir: str):
    def parse(raw):
        line = json.loads(raw.decode())
        #   at this step you need to limit the length which you design
        #   so when use padded_batch func the whole length is the fixed length
        #   padded_shapes None None (1000, 25)
        contents = line['content'].split()[:1000]
        titles = line['title'].split()[:25]
        return contents, titles

    file_lists = tf.gfile.Glob(os.path.join(data_dir, '*' + str(mode) + '*'))
    ds = tf.data.TextLineDataset(file_lists)
    ds.repeat(None).map(lambda line: tf.py_func(parse, [line], Tout=[tf.string, tf.string])).map(
        lambda x, y: ({'feature': x}, y)).padded_batch(batch_size=2, padded_shapes=({'feature': [None]}, [None]),
                                                       padding_values=({'feature': '<PAD>'}, '<PAD>'))

    return ds
