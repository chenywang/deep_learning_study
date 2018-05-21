# View more python tutorial on my Youtube and Youku channel!!!

# Youtube video tutorial: https://www.youtube.com/channel/UCdyjiB5H8Pu7aDTNVXTTpcg
# Youku video tutorial: http://i.youku.com/pythontutorial

"""
Please note, this code is only for python 3+. If you are using python 2+, please modify the code accordingly.
"""

import numpy as np
import tensorflow as tf

# create data
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data * 0.5 + 0.3

# create tensorflow structure start ###
Weights = tf.Variable(tf.random_uniform([1], -1.0, 1.0))

biases = tf.Variable(tf.zeros([1]))

y = Weights * x_data + biases

loss = tf.reduce_mean(tf.square(y - y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)
print("zeros type", type(tf.zeros([1])))
print("Weights type", type((Weights)))
print("y type", type(y))
print("loss type", type(y))
print("train type", type(train))
### create tensorflow structure end ###

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

for step in range(201):
    train_result = sess.run(train)
    if step % 20 == 0:
        print step, sess.run(Weights), sess.run(biases)
