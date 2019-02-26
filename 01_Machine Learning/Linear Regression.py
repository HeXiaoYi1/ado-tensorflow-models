# -*- coding:utf-8 -*-
# author: adowu

"""
A linear regression learning algorithm example using Tensorflow
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as pt

# Training Data
train_X = np.asarray([3.3, 4.4, 5.5, 6.71, 6.93, 4.168, 9.779, 6.182, 7.59, 2.167,
                      7.042, 10.791, 5.313, 7.997, 5.654, 9.27, 3.1])
train_Y = np.asarray([1.7, 2.76, 2.09, 3.19, 1.694, 1.573, 3.366, 2.596, 2.53, 1.221,
                      2.827, 3.465, 1.65, 2.904, 2.42, 2.94, 1.3])
#
assert train_X.shape[0] == train_Y.shape[0]

n_samples = train_X.shape[0]
X = tf.placeholder(dtype='float', shape=None, name='X')
Y = tf.placeholder(dtype='float', shape=None, name='Y')

#   standard normal distribution for randn
#   Variable's trainable = True
W = tf.Variable(np.random.randn(), name='weights')
b = tf.Variable(np.random.randn(), name='bias')

pred = tf.add(tf.multiply(X, W), b)

#   Mean Square Error
cost = tf.reduce_sum(tf.pow(pred - Y, 2)) / (2 * n_samples)

#   Gradient Descent
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

#   initialize the variables (assign their default value)
init = tf.global_variables_initializer()

with tf.Session() as sess:
    #   run the initializer
    sess.run(init)
    #   -1.2221383
    print(sess.run(W))
    #   1.3165525
    print(sess.run(b))

    for epoch in range(1000):
        for (x, y) in zip(train_X, train_Y):
            sess.run(optimizer, feed_dict={X: x, Y: y})

        if (epoch + 1) % 50 == 0:
            c = sess.run(cost, feed_dict={X: train_X, Y: train_Y})
            print("Epoch:", '%05d' % (epoch + 1), "cost=", "{:.4f}".format(c), "W=", sess.run(W), "b=", sess.run(b))

    print('Optimizer Finished')
    train_cost = sess.run(cost, feed_dict={X: train_X, Y: train_Y})
    print("Training cost=", train_cost, "W=", sess.run(W), "b=", sess.run(b), '\n')

    # graphic display
    pt.plot(train_X, train_Y,'ro', label='Original Data')
    pt.plot(train_X, sess.run(W) * train_X + sess.run(b), label='Fitted Line')
    pt.legend()
    pt.show()

    # Testing example
    test_X = np.asarray([6.83, 4.668, 8.9, 7.91, 5.7, 8.7, 3.1, 2.1])
    test_Y = np.asarray([1.84, 2.273, 3.2, 2.831, 2.92, 3.24, 1.35, 1.03])

    print("Testing... (Mean square loss Comparison)")
    testing_cost = sess.run(
        tf.reduce_sum(tf.pow(pred - Y, 2)) / (2 * test_X.shape[0]),
        feed_dict={X: test_X, Y: test_Y})
    print("Testing cost=", testing_cost)
    print("Absolute mean square loss difference:", abs(
        train_cost - testing_cost))

    pt.plot(test_X, test_Y, 'bo', label='Testing data')
    pt.plot(train_X, sess.run(W) * train_X + sess.run(b), label='Fitted line')
    pt.legend()
    pt.show()

