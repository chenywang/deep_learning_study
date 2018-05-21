# -*- coding: UTF-8 -*-

# View more python learning tutorial on my Youtube and Youku channel!!!

# Youtube video tutorial: https://www.youtube.com/channel/UCdyjiB5H8Pu7aDTNVXTTpcg
# Youku video tutorial: http://i.youku.com/pythontutorial

"""
Please note, this code is only for python 3+. If you are using python 2+, please modify the code accordingly.
"""
import tensorflow as tf
import numpy as np

state = tf.Variable(0, name='counter')
one = tf.constant(1)
# one = tf.Variable(1, name='one')
new_value = tf.add(state, one)
update = tf.assign(state, new_value)
Weights = tf.Variable(tf.random_uniform([5, 5], -1.0, 1.0))
Weights2 = tf.Variable(tf.random_uniform([5, 5], 0, 1.0))

# zeros = tf.zeros([3, 4], tf.int32)  # [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
zeros = tf.Variable(tf.zeros([1, 10]) + 0.1)


# 使用默认的session的方法
sess2 = tf.InteractiveSession()
init = tf.global_variables_initializer()
sess2.run(init)

print "cool", Weights.eval()
with tf.Session() as sess:
    sess.run(init)
    print "weight:", sess.run(Weights)
    print "weight2:", sess.run(Weights2)
    print "zeros", sess.run(zeros)
    for _ in range(3):
        print "new_value:", sess.run(new_value)
        print "update:", sess.run(update)
        print "state:", sess.run(state)

