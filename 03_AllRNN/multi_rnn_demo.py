# -*- coding:utf-8 -*-
# author: adowu
import tensorflow as tf

tf.enable_eager_execution()


def multi_rnn_demo():
    #   cell = tf.nn.rnn_cell.BasicRNNCell(num_units=6)
    #   m_cell = tf.nn.rnn_cell.MultiRNNCell([cell] * 3)
    #   这样定义的话是错误的，相当于是用一个cell，这样在第一层的时候，输入是[2,4](这时候的kernel_=[10,6])，输出是[2,6]
    #   gate_inputs = math_ops.matmul(array_ops.concat([inputs, state], 1), self._kernel)
    #   到第二层的时候输入是：[2,6],如果是用一个cell的话，这时候的kernel_=[10,6]
    #   而inputs和state的shape都为[2,6],这样做matmul就有问题了。
    #   所以在定义多层cells的时候一定要注意了每一个cell都应该是不同的cell。
    #   这也就是为什么一些blog说embedding_size和hidden_size要是一样的原因了。
    #   只要定义对了cell，维度当然是可以不同的了。
    m_cell = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.BasicRNNCell(num_units=3 * i) for i in range(1, 5)])
    wrapper = tf.nn.rnn_cell.DropoutWrapper(m_cell, input_keep_prob=0.5, output_keep_prob=0.8)
    a = tf.random_normal([2, 4])
    outs, states = wrapper(
        inputs=a,
        state=m_cell.zero_state(2, dtype=tf.float32)
    )
    """
    维度是[2,12],也就是最后一层BasicRNNCell(3*4)的维度
    tf.Tensor(
        [[-0.46235284 -0.13470592 -0.23693396 -0.364829    0.27304482  0.37045002
           0.02142742  0.05885093 -0.22207676 -0.164027    0.36846045  0.        ]
         [ 0.11636553  0.          0.          0.35240266 -0.30226937 -0.00565579
          -0.13215029  0.         -0.00331525  0.04132571 -0.          0.        ]], shape=(2, 12), dtype=float32)
    
    states输出是一个tuple，对应下标表示对应层的BasicRNNCell(3*i)的输出结果，如果是LSTMCell的话，每一层还有c和h
    (
        <tf.Tensor: id=80, shape=(2, 3), dtype=float32, numpy=
            array([[ 0.50330746, -0.64860016, -0.7912944 ],
                   [-0.8363293 ,  0.12551163, -0.33754644]], dtype=float32)>, 
        <tf.Tensor: id=111, shape=(2, 6), dtype=float32, numpy=
            array([[ 0.60967964,  0.3295311 ,  0.10356555,  0.69681686,  0.1016121 ,-0.40134493],
                   [-0.02086192,  0.53273064,  0.0425563 , -0.39393067, -0.2405967 ,0.0720734 ]], dtype=float32)>, 
        <tf.Tensor: id=142, shape=(2, 9), dtype=float32, numpy=
            array([[-0.14141226,  0.1055992 ,  0.25089046,  0.07378653, -0.42437685,
                     0.4792934 , -0.27223024, -0.39165986, -0.02905593],
                   [ 0.30640686, -0.01233759,  0.24259914,  0.34790638, -0.00120982,
                     0.02672288,  0.11938408,  0.3234135 ,  0.14079471]],
                  dtype=float32)>, 
        <tf.Tensor: id=173, shape=(2, 12), dtype=float32, numpy=
            array([[-0.3698823 , -0.10776474, -0.18954717, -0.2918632 ,  0.21843587,
                     0.29636002,  0.01714193,  0.04708075, -0.1776614 , -0.1312216 ,
                     0.29476836,  0.09352753],
                   [ 0.09309243,  0.14289957,  0.02828147,  0.28192213, -0.2418155 ,
                    -0.00452463, -0.10572024,  0.15910426, -0.0026522 ,  0.03306057,
                    -0.2773368 ,  0.18081182]], dtype=float32)>)
    """
    print(outs)
    print(states)


def multi_rnn_demo_02():
    m_cell = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.BasicRNNCell(num_units=3 * i) for i in range(1, 5)])
    wrapper = tf.nn.rnn_cell.DropoutWrapper(m_cell, input_keep_prob=0.5, output_keep_prob=0.8)
    a = tf.random_normal([2, 3, 4])
    state = m_cell.zero_state(2, tf.float32)
    output, state = tf.nn.dynamic_rnn(wrapper, a, initial_state=state, time_major=False)
    print(output)
    """
    (
    <tf.Tensor: id=324, shape=(2, 3), dtype=float32, numpy=
        array([[ 0.04508227, -0.23633523,  0.30065763],
                [ 0.7612733 , -0.52450776, -0.6661888 ]], dtype=float32)>, 
    <tf.Tensor: id=331, shape=(2, 6), dtype=float32, numpy=
        array([[ 0.54584306, -0.17747684, -0.44227242,  0.2842951 ,  0.43393528,-0.63828385],
               [-0.5828087 , -0.5281809 ,  0.06895413, -0.74510646, -0.53972644,
                 0.15232728]], dtype=float32)>, 
    <tf.Tensor: id=338, shape=(2, 9), dtype=float32, numpy=
        array([[ 0.5603744 , -0.04951737, -0.26185265,  0.08209028,  0.672668  ,
                0.18829748,  0.27329427,  0.4323986 ,  0.09414196],
            [ 0.03460297, -0.20349553,  0.47373882,  0.09012283,  0.02843269,
            -0.6374881 , -0.10102204, -0.16552992,  0.35916832]],dtype=float32)>, 
    <tf.Tensor: id=345, shape=(2, 12), dtype=float32, numpy=
            array([[ 0.0955615 , -0.17922615, -0.3616308 ,  0.23105423, -0.12791048,
                     0.12003306,  0.07470868, -0.50892484,  0.04814089, -0.22584775,
                     0.48966137, -0.0903751 ],
                   [-0.1615388 , -0.00884357,  0.03238142,  0.1927497 ,  0.18858932,
                     0.09198219, -0.04517724, -0.00153504,  0.00972735, -0.35469332,
                     0.25491163,  0.24003494]], dtype=float32)>)
    """
    print(state)


def dropout_test():
    """
    tf.Tensor(
        [[ 0.563435   -1.9721379   1.5006789  -2.2403116 ]
         [-0.39942354 -0.14369683  2.0294921  -1.0674685 ]], shape=(2, 4), dtype=float32)
    tf.Tensor(
        [[ 0.        -0.         0.        -0.       ]
         [-0.7988471 -0.         4.0589843 -0.       ]], shape=(2, 4), dtype=float32)
    """
    a = tf.random_normal(shape=[2, 4])
    #   ret = math_ops.div(x, keep_prob) * binary_tensor，会看到输出的值是原来的1/0.5 倍，深度学习中正是这样解决，训练和预测过程的dropout
    #   并不是说每一个的输入中有一半的输入会为0
    b = tf.nn.dropout(x=a, keep_prob=0.5)
    print(a)
    print(b)


if __name__ == '__main__':
    multi_rnn_demo_02()
