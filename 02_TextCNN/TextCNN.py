# -*- coding:utf-8 -*-
# author: adowu
import tensorflow as tf


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
        h_pool = tf.reshape(h_pool, [-1, params.get('filters') * len(params.get('kernels'))])
        if is_training:
            h_pool = tf.layers.dropout(h_pool, rate=params.get('dropout'))

        logits = tf.layers.dense(h_pool, params.get('num_class'))

        return logits


def model_fn(features, labels, mode, params: dict):
    x = features
    init_embeddings = tf.random_uniform([params.get('vocab_size'), params.get('embedding_size')])
    embeddings = tf.get_variable('embeddings', initializer=init_embeddings)
    x = tf.nn.embedding_lookup(embeddings, x)

    logits = textCNNNet(x, params, mode == tf.estimator.ModeKeys.TRAIN)

    if mode == tf.estimator.ModeKeys.PREDICT:
        estimator_spec = tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=logits
        )
        return estimator_spec

    pred_class = tf.argmax(logits, axis=1)
    loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        labels=labels,
        logits=logits
    ))

    optimizer = tf.train.AdamOptimizer(learning_rate=params.get('learning_rate'))

    train_op = optimizer.minimize(loss_op, global_step=tf.train.get_global_step())

    acc_op = tf.metrics.accuracy(labels=labels, predictions=pred_class)

    estimator_spec = tf.estimator.EstimatorSpec(
        mode,
        predictions=pred_class,
        loss=loss_op,
        train_op=train_op,
        eval_metric_ops={'accuracy': acc_op}
    )

    return estimator_spec
