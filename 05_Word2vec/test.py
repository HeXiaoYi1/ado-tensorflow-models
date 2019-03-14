#-*- coding:utf-8 -*-
# author: adowu

import tensorflow as tf
tf.enable_eager_execution()

a = tf.constant(value=[[[1,2,3,4],[1,2,3,4],[1,2,3,4]],[[2,3,4,5],[2,3,4,5],[2,3,4,5]]])
b = tf.reduce_mean(a, axis=1)
c = tf.reduce_sum(a, axis=2)

print(a)
print(b)
print(c)