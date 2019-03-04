# -*- coding:utf-8 -*-
# author: adowu
import tensorflow as tf

tf.enable_eager_execution()


def basic_lstm_demo():
    """
        结合着LSTM的图示来理解代码更清楚。

        #   输入的inputs [2,3,4],经过unstack则为 list([2,4]).size为3，所以输入到LSTM中的input为[2,4]
        #   初始化的 c 和 h 都是zero_state 也就是都为[2,4]的zero，这是参数state_is_tuple的情况下，
        #   如果这个参数为 False，则 c,h = [2,2]
        c, h = state
        #   初始化权重参数为：在此处就是 [4 + 4, 4 * 4] = [8, 16],为什么乘以4后面就可以知道原因
        kernel_ = [input_dims +num_units, 4 * num_units]

        #   concat[inputs, h] = [2, 8] kernel_ = [8, 16], bias=zero of [4 * num_units]
        #   所以gate_inputs = [2, 16]
        gate_inputs = bias_add(matmul(concat([inputs, h], axis=1), kernel_)， bias)

        #   i 表示input_gate
        #   j 表示new_input
        #   f 表示forget_gate
        #   o 表示output_gate
        #   为了保持维度正确，所以前面要在num_units上乘以4的原因
        i, j, f, o = array_ops.split(value=gate_inputs, num_or_size_splits=4, axis=1)

        forget_bias = 1.0

        #   计算这个cell中的new_c 和 new_h
        #   forget_gate_output =  sigmoid(add(f, forget_bias_tensor))
        #   input_gate_output = multiply(sigmoid(i), tanh(j))
        #   update_c = add(multiply(c, forget_gate_output), input_gate_output)
        #   output_gate_output = multiply(tanh(new_c), sigmoid(o))

        new_c = add(multiply(c, sigmoid(add(f, forget_bias_tensor))),multiply(sigmoid(i), tanh(j)))
        new_h = multiply(tanh(new_c), sigmoid(o))

    """
    cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=4)
    tf.nn.rnn_cell.LSTMCell()
    zero_state = cell.zero_state(batch_size=2, dtype=tf.float32)
    a = tf.random_normal([2, 3, 4])

    out, state = tf.nn.dynamic_rnn(
        cell=cell,
        initial_state=zero_state,
        inputs=a
    )
    """
    每一个时间步的输出:shape 为 [2,3,4]
    tf.Tensor(
        [[[ 0.29594404 -0.06257749  0.00272913  0.38393494]
          [ 0.12317018 -0.10669467  0.21305212 -0.0534559 ]
          [ 0.11735746 -0.03012969  0.08865868 -0.10764799]]
        
         [[-0.07051807  0.02736617  0.07237878 -0.19151129]
          [-0.07522646  0.00569247 -0.01109379 -0.00774325]
          [ 0.05763769 -0.00310471  0.21375947 -0.16625713]]], shape=(2, 3, 4), dtype=float32)
  
    最后一个时间步的输出，包括c 和 h shape 都为 [2,4]
    LSTMStateTuple(
            c=<tf.Tensor: id=309, shape=(2, 4), dtype=float32, numpy=
            array([[ 0.26399267, -0.09096628,  0.1642536 , -0.30149382],
                   [ 0.2447102 , -0.00411555,  0.38746575, -0.21990177]],
                  dtype=float32)>, 
            h=<tf.Tensor: id=312, shape=(2, 4), dtype=float32, numpy=
            array([[ 0.11735746, -0.03012969,  0.08865868, -0.10764799],
                    [ 0.05763769, -0.00310471,  0.21375947, -0.16625713]],
                    dtype=float32)>
            )
    """

    print(out)
    print(state)



def lstm_test():
    a = tf.random_normal([2, 3, 4])
if __name__ == '__main__':
    basic_lstm_demo()
