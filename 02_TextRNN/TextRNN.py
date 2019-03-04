# -*- coding:utf-8 -*-
# author: adowu
import tensorflow as tf


class TextRNNLayer(tf.layers.Layer):
    def __init__(self, hidden_size, dropout):
        super(TextRNNLayer, self).__init__()


        self.fw_cell = tf.nn.rnn_cell.LSTMCell(num_units=hidden_size)
        self.bw_cell = tf.nn.rnn_cell.LSTMCell(num_units=hidden_size)
        if dropout is not None:
            self.fw_cell = tf.nn.rnn_cell.DropoutWrapper(cell=self.fw_cell, input_keep_prob=dropout)
            self.bw_cell = tf.nn.rnn_cell.DropoutWrapper(cell=self.bw_cell, input_keep_prob=dropout)

        self.initializer = tf.random_normal_initializer(stddev=0.1)

    def call(self, inputs, **kwargs):
        #   hyper parameters
        params = kwargs.get('params')
        embeddings = tf.get_variable(
            name='embeddings',
            shape=[params.get('vocab_size'), params.get('embedding_size')],
            initializer=self.initializer
        )
        #   [batch_size, sequence_max_length] -> [batch_size, sequence_max_length,embedding_size]
        feature = tf.nn.embedding_lookup(embeddings, inputs)
        """
        output = (fw_output, bw_output) each shape is [batch_size, sequence_max_length,hidden_size]
        output_state = (fw_output_state, bw_output_state) each shape is [batch_size, hidden_size]
        """
        output, output_state = tf.nn.bidirectional_dynamic_rnn(
            cell_fw=self.fw_cell,
            cell_bw=self.bw_cell,
            inputs=feature,
            dtype=tf.float32
        )
        #   [batch_size, sequence_max_length,hidden_size*2]
        output = tf.concat(output, axis=2)
        #   [batch_size, hidden_size*2]
        #   there are two ways to get final out
        #   rnn_out = tf.reduce_mean(output, axis=1)
        rnn_out = output[:, -1, :]
        #   [batch_size, num_classes]
        logits = tf.layers.dense(inputs=rnn_out, units=params.get('num_class'))

        return logits


def model_test():
    layer = TextRNNLayer(8, 0.5)
    a = tf.random_uniform(shape=[2, 3], dtype=tf.int32, minval=1, maxval=10)
    params = dict()
    params['vocab_size'] = 10
    params['embedding_size'] = 10
    params['num_classes'] = 3
    labels = list()
    labels.append(1)
    logits = layer(a, params=params)
    print(logits)


if __name__ == '__main__':
    model_test()

