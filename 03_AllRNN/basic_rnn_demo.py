# -*- coding:utf-8 -*-
# author: adowu
import tensorflow as tf

tf.enable_eager_execution()

def basic_rnn_demo():
    """
    Most basic rnn
    tanh(W * input + U * state + B)

    #   inputs [2,3,4]  state = [2,4]
    #   unstack(inputs) = 3 size [2,4]
    #   每次是进行一个batch的时间步骤的计算，第一个进行的就是每个batch中的第一个字
    #   a = [2, 8]
    a = concat([inputs,state], 1)
    #   kernel_ = [8, 4] kernel_在每个时间步都是共享的
    kernel_ = [inputs_dim + num_units,num_units]
    #   b = [2, 4]
    b = matmul(a, kernel_)
    #   c = [2, 4] bias初始化为0 bias 在每个时间步都是共享的
    c = b + bias
    #   [2, 4]
    #   会返回一个tuple，内容都是output，一个作为此时刻的state,这样state就可以了从第一个一直往后更新
    output,output = tanh(c)
    """


    cell = tf.nn.rnn_cell.BasicRNNCell(num_units=4)
    zero_state = cell.zero_state(batch_size=2, dtype=tf.float32)
    a = tf.random_normal([2, 3, 4])

    """
    out
    tf.Tensor(
    [[[ 0.7875833   0.11634824  0.31249827  0.11648687]
      [ 0.6418752  -0.9281747   0.6534868   0.3821376 ]
      [ 0.9750985  -0.40439364  0.9770327   0.8529797 ]]
    
     [[-0.09945039 -0.49678802 -0.32603818  0.20098403]
      [-0.57557577  0.15389016 -0.7197561  -0.36572933]
      [ 0.4485007  -0.51780844 -0.6015551   0.16041796]]], shape=(2, 3, 4), dtype=float32)
    
    state
    tf.Tensor(
    [[ 0.9750985  -0.40439364  0.9770327   0.8529797 ]
     [ 0.4485007  -0.51780844 -0.6015551   0.16041796]], shape=(2, 4), dtype=float32)
    """
    out, state = tf.nn.dynamic_rnn(
        cell=cell,
        initial_state=zero_state,
        inputs=a
    )

    print(out)
    print(state)


if __name__ == '__main__':
    basic_rnn_demo()
