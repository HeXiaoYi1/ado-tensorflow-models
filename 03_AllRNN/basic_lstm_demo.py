# -*- coding:utf-8 -*-
# author: adowu
import tensorflow as tf

tf.enable_eager_execution()

def basic_lstm_demo():


    cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=4)
    zero_state = cell.zero_state(batch_size=2, dtype=tf.float32)
    a = tf.random_normal([2, 3, 4])

    out, state = tf.nn.dynamic_rnn(
        cell=cell,
        initial_state=zero_state,
        inputs=a
    )

    print(out)
    print(state)


if __name__ == '__main__':
    basic_lstm_demo()
