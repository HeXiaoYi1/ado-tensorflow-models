# -*- coding:utf-8 -*-
# author: adowu
import tensorflow as tf

tf.enable_eager_execution()
"""
BahdanauAttention:
https://arxiv.org/abs/1409.0473
"""


def attention(inputs, hidden_size, dropout, attention_size):
    """
    :param inputs: [B, T, D] -> [batch_size, sequence_length, embedding_size]
    :param hidden_size: RNN output size
    :param dropout: dropout rate
    :param attention_size: attention output size
    :return:
    """
    fw = tf.nn.rnn_cell.GRUCell(hidden_size, name='fw')
    bw = tf.nn.rnn_cell.GRUCell(hidden_size, name='bw')

    if dropout:
        fw = tf.nn.rnn_cell.DropoutWrapper(fw, output_keep_prob=dropout)
        bw = tf.nn.rnn_cell.DropoutWrapper(bw, output_keep_prob=dropout)

    output, _ = tf.nn.bidirectional_dynamic_rnn(
        fw,
        bw,
        inputs=inputs,
        dtype=tf.float32
    )

    #   [batch_size, sequence_length, 2 * hidden_size]
    output = tf.concat(output, axis=2)

    #   W * X + B
    #   [batch_size, sequence_length, 2 * hidden_size] -> [batch_size, sequence_length, attention_size]
    I = tf.layers.dense(inputs=output, units=attention_size, activation=tf.tanh)

    V = tf.get_variable(name='v_omega', shape=[attention_size], dtype=tf.float32)

    #   [batch_size, sequence_length, attention_size]
    U = tf.multiply(I, V)
    #   [batch_size, sequence_length]
    U = tf.reduce_sum(U, axis=2)
    #   [batch_size, sequence_length]
    A = tf.nn.softmax(U, axis=1)

    #   multiply is [batch_size, sequence_length, 2 * hidden_size] * [batch_size, sequence_length, 1]
    #   multiply = [batch_size, sequence_length, 2 * hidden_size]
    #   reduce_sum = [batch_size, 2 * hidden_size]
    tmp = tf.multiply(output, tf.expand_dims(A, -1))
    C = tf.reduce_sum(tf.multiply(output, tf.expand_dims(A, -1)), axis=1)
    # C = tf.reduce_mean(tf.multiply(output, tf.expand_dims(A, -1)), axis=1)

    """
    A
    tf.Tensor(
        [[0.28975958 0.35240924 0.35783118]
         [0.23674823 0.45426124 0.30899054]], shape=(2, 3), dtype=float32)
         
    C
    tf.Tensor(
        [[ 0.12757744  0.26259103 -0.2159944  -0.00717294  0.          0.
           0.16267745  0.          0.13928103  0.          0.12901253  0.10434134]
         [ 0.14803177  0.         -0.08678789  0.29228723 -0.22902207  0.11991494
           0.10608757  0.13675767  0.11566126  0.24621248  0.10165013  0.00688825]], shape=(2, 12), dtype=float32)
    """
    return C, A


if __name__ == '__main__':
    a = tf.random_normal(shape=[2, 3, 4])

    C, A = attention(a, 6, 0.5, 8)
    print(C)
    print(A)
