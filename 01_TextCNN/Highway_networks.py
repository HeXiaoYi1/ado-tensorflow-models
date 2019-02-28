# -*- coding:utf-8 -*-
# author: adowu
import tensorflow as tf


def highwayNet(feature, output_size, num_layer=2, bias=-1.0):

    def linear(feature, output_size):
        return tf.layers.dense(
            inputs=feature,
            units=output_size,
            activation=tf.nn.relu
        )

    output = feature
    for each in range(num_layer):
        H = tf.nn.relu(linear(output, output_size), name='activation')
        T = tf.nn.sigmoid(linear(output, output_size) + bias, name='transform_gate')
        # T_1 = tf.nn.sigmoid(linear(output, output_size), name='transform_gate')
        """
            print(T)
                tf.Tensor(
                        [[0.48157623 0.26894143 0.26894143 0.26894143 0.46048874 0.26894143 0.27552864 0.26894143]
                         [0.368299   0.26894143 0.26894143 0.26894143 0.59945595 0.35760435 0.3215622  0.26894143]], 
                         shape=(2, 8), dtype=float32)
                         
            print(T_1)
                tf.Tensor(
                        [[0.55119145 0.5438615  0.5        0.5        0.6005574  0.52241236 0.5        0.6611666 ]
                         [0.5        0.5279554  0.5        0.5365138  0.5        0.5 0.5        0.80558586]], 
                         shape=(2, 8), dtype=float32)
                         
            初始化的bias定义为 negative，论文中给的是 -1 -3等，从上面输出也能看出，在前期网络更侧重于搬运行为，
            T 越小 1 - T 就越大，也就是 C 越大
        """
        C = tf.subtract(1., T, name='carry_gate')
        output = H * T + C * output

    return output


if __name__ == '__main__':
    tf.enable_eager_execution()
    feature = tf.random_uniform(shape=[2, 8])
    """
    print(feature)
        tf.Tensor(
        [[0.12734997 0.5313252  0.11216724 0.8434391  0.558702   0.85594463 0.55231774 0.02937436]
         [0.04293442 0.5114585  0.7133622  0.1873641  0.95842624 0.27655768 0.35945523 0.8060981 ]], 
         shape=(2, 8), dtype=float32)
    """

    output_size = feature.get_shape().as_list()[1]
    """
    print(output_size)
        8
    """

    """
        tf.Tensor(
        [[0.1623639  0.2855093  0.13142484 0.6306237  0.45616442 0.3370946 0.2863167  0.11120965]
         [0.13969976 0.2733473  0.51078534 0.3886435  0.52740085 0.1193624 0.37235203 0.47573265]], 
         shape=(2, 8), dtype=float32)
    """
    print(highwayNet(feature, output_size))
