# -*- coding:utf-8 -*-
# author: adowu
from TextRNN import TextRNNLayer
import tensorflow as tf


def model_fn(features, labels, mode, params: dict):
    rnn = TextRNNLayer(params.get('hidden_size'), params.get('drop_out'))
    logits = rnn(features, params=params)
    probs = tf.nn.softmax(logits)
    predictions = tf.argmax(probs, axis=1)
    output = {
        'predictions': predictions,
        'probabilities': probs
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=output,
            export_outputs={'output': tf.estimator.export.PredictOutput(output)}
        )

    loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels,
        logits=logits
    ))

    optimizer = tf.train.AdamOptimizer(learning_rate=params.get('learning_rate'))

    train_op = optimizer.minimize(loss_op, global_step=tf.train.get_global_step())
    accuracy = tf.metrics.accuracy(labels=labels, predictions=predictions)

    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        loss=loss_op,
        train_op=train_op,
        eval_metric_ops={'accuracy': accuracy}
    )
