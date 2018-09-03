# -*- coding:utf-8 -*-
"""
@contact: adonis_wu@outlook.com
@file: ado_transform.py
@time: 2018/9/3 16:49

step one:   transform metadata to tf records
step two:   read tf records batch
step three: truncate or pad each batch data

"""
__author__ = '🍊 Adonis Wu 🍊'

import tensorflow as tf
from tensorflow.python.lib.io import file_io


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _int64_list(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def transform_records_v1():
    save_path = 'tmp.tfrecords'
    label = '0'
    a_sentences = list()
    w_sentences = list()

    with tf.python_io.TFRecordWriter(save_path) as writer:
        labels = {
            'label': _bytes_feature([str(label).encode()])
        }
        feature = {
            'chars': tf.train.FeatureList(
                feature=[_bytes_feature([str(char).encode() for char in sentence]) for sentence in a_sentences]
            ),
            'words': tf.train.FeatureList(
                feature=[_bytes_feature([str(word).encode() for word in sentence]) for sentence in w_sentences]
            ),
        }

        example = tf.train.SequenceExample(
            feature_lists=tf.train.FeatureLists(feature_list=feature),
            context=tf.train.Features(feature=labels)
        )

        writer.write(example.SerializeToString())


def read_records_v1(mode: tf.estimator.ModeKeys, data_path, params):
    sequence_features = {
        'words': tf.VarLenFeature(tf.string),
        'chars': tf.VarLenFeature(tf.string),
    }

    label_features = {
        'label': tf.FixedLenFeature([], tf.string)
    }

    def parse(raw):
        context_dict, sequence_dict = tf.parse_single_sequence_example(
            serialized=raw,
            context_features=label_features,
            sequence_features=sequence_features
        )

        features = {
            'chars': sequence_dict.get('chars'),
            'words': sequence_dict.get('words')
        }

        labels = {
            'label': context_dict.get('label')
        }

        if mode == tf.estimator.ModeKeys.PREDICT:
            return features

        return features, labels

    file_path = file_io.get_matching_files(data_path)
    ds = tf.data.TFRecordDataset(file_path, buffer_size=30 * 1024 * 1024, num_parallel_reads=2)

    if mode == tf.estimator.ModeKeys.TRAIN:
        result = ds.repeat(None).shuffle(buffer_size=1024 * 1024).map(parse).batch(params.batch_size).prefetch(
            buffer_size=None)
    elif mode == tf.estimator.ModeKeys.EVAL:
        result = ds.repeat(None).take(params.all_eval_size).map(parse).batch(params.batch_size).prefetch(
            buffer_size=None)
    else:
        result = ds.repeat(None).take(params.all_predict_size).map(parse).batch(params.batch_size).prefetch(
            buffer_size=None)
    #   result is sparse tensor when you train the data you should use below function get_dense_input transform to dense
    return result


def get_dense_input(input, params):
    """
              if you use export_savedmodel which input is the get_input_receiver_fn() and this is a dense tensor,
              also it's mode is PREDICT too,
              so that at this time you can not use sparse_tensor_to_dense
              when you use estimator.predict() function the input is a sparse which is need to dense
              this is the conflict
    """
    # if mode != tf.estimator.ModeKeys.PREDICT:
    #     #   sparse tensor   ->  dense tensor
    #     dense_input = tf.sparse_tensor_to_dense(input, default_value=0)
    # else:
    #     dense_input = input

    dense_input = tf.sparse_tensor_to_dense(input, default_value=0)
    shape = tf.shape(dense_input)
    batch_size = tf.gather(shape, 0)
    current_sentence_num = tf.gather(shape, 1)
    current_sequence_length = tf.gather(shape, 2)

    def sentence_truncate():
        #   get batch train data
        result = tf.slice(dense_input, [0, 0, 0],
                          [batch_size, tf.constant(params.sentence_num), current_sequence_length])

        return result

    def sentence_padding():
        #   get batch train data
        result = tf.concat([dense_input,
                            tf.zeros([batch_size,
                                      tf.subtract(tf.constant(params.sentence_num), current_sentence_num),
                                      current_sequence_length],
                                     dtype=tf.int64)], axis=1)
        return result

    #   if current_sentence_num >= params.sentence_num then doc_truncate else doc_padding
    dense_input = tf.cond(
        tf.greater_equal(current_sentence_num, tf.constant(params.sentence_num)),
        sentence_truncate,
        sentence_padding
    )

    def sequence_truncate():
        result = tf.slice(dense_input, [0, 0, 0], [-1, -1, params.sequence_length])
        return result

    def sequence_padding():
        result = tf.concat(
            [dense_input, tf.zeros([
                batch_size, params.sentence_num, tf.subtract(tf.constant(params.sequence_length),
                                                             current_sequence_length)], dtype=tf.int64)],
            axis=2
        )
        return result

    dense_input = tf.cond(
        tf.greater_equal(current_sequence_length, tf.constant(params.sequence_length)),
        sequence_truncate,
        sequence_padding
    )

    return dense_input
