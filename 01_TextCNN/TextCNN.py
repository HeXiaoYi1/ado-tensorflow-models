# -*- coding:utf-8 -*-
# author: adowu
import tensorflow as tf
from Highway_networks import highwayNet


def textCNNNet(features, params: dict, is_training):
    with tf.variable_scope('TextCNNNet'):
        #   inputs = [batch_size, sequence_length, embedding_size]
        x = features
        x = tf.expand_dims(x, -1)
        pools = []
        for kernel in params.get('kernels'):
            #   [batch_size,sequence_max_length - kernel + 1, 1, filters]
            conv = tf.layers.conv2d(
                inputs=x,
                filters=params.get('filters'),
                kernel_size=[kernel, params.get('embedding_size')],
                strides=1,
                padding='valid',
                activation=tf.nn.relu
            )
            #   [batch_size, 1, 1, filters]
            pool = tf.layers.max_pooling2d(
                inputs=conv,
                pool_size=[params.get('sequence_max_length') - kernel + 1, 1],
                strides=1,
                padding='valid'
            )

            pools.append(pool)

        h_pool = tf.concat(pools, axis=3)
        total_filters = params.get('filters') * len(params.get('kernels'))
        h_pool = tf.reshape(h_pool, [-1, total_filters])
        if is_training:
            h_pool = tf.layers.dropout(h_pool, rate=params.get('dropout'))
        if params.get('use_highway', False):
            #   considering add HighwayNet
            h_pool = highwayNet(h_pool, total_filters)
        logits = tf.layers.dense(h_pool, params.get('num_class'))

        return logits


def model_fn(features, labels, mode, params: dict):
    x = features
    init_embeddings = tf.random_uniform([params.get('vocab_size'), params.get('embedding_size')], -1., 1.)
    embeddings = tf.get_variable('embeddings', initializer=init_embeddings)
    x = tf.nn.embedding_lookup(embeddings, x)

    logits = textCNNNet(x, params, mode == tf.estimator.ModeKeys.TRAIN)
    probs = tf.nn.softmax(logits)
    predictions = tf.argmax(logits, axis=1, name='predictions')
    output = {
        'predictions': predictions,
        'probabilities': probs
    }
    #   TODO 目前不是很明白 这样定义的用途在哪里，predict结果中似乎也无法获取到
    #   说明： export_outputs 指的是在模型导出的时候export_savedmodel(),这样调用模型分析的时候就可以了通过字段获取到对应结果了
    if mode == tf.estimator.ModeKeys.PREDICT:
        estimator_spec = tf.estimator.EstimatorSpec(
            mode=mode,
            #   TODO 这个地方的predictions，既可以是argmax之后的predictions，也可以是softmax之后的probs，并没有找到同时返回的方法
            #   解决了同时返回的方法，其实predictions是可以接收一个dict的，在里面加上自己想返回的结果就可以了
            #   后面再尝试export等相关功能，可参考法研杯的相关代码
            predictions=output,
            export_outputs={'output': tf.estimator.export.PredictOutput(output)}
        )
        return estimator_spec

    loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels,
        logits=logits
    ))

    optimizer = tf.train.AdamOptimizer(learning_rate=params.get('learning_rate'))

    train_op = optimizer.minimize(loss_op, global_step=tf.train.get_global_step())
    #   TODO 增加更多新的评价指标
    acc_op = tf.metrics.accuracy(labels=labels, predictions=predictions)

    estimator_spec = tf.estimator.EstimatorSpec(
        mode,
        predictions=predictions,
        loss=loss_op,
        train_op=train_op,
        eval_metric_ops={'accuracy': acc_op}
    )

    return estimator_spec
