import numpy as np
import tensorflow as tf


def linear_regression():
    # prepared train data
    M = 100
    N = 2
    w_data = np.mat([[1.0, 3.0]]).T
    b_data = 10
    x_data = np.random.randn(M, N).astype(np.float32)
    y_data = np.mat(x_data) * w_data + 10 + np.random.randn(M, 1) * 0.33

    # define model and graph
    w = tf.Variable(tf.random_uniform([N, 1], -1, 1))
    b = tf.Variable(tf.random_uniform([1], -1, 1))
    y = tf.matmul(x_data, w) + b
    loss = tf.reduce_mean(tf.square(y - y_data))

    # choose optimizer
    train_op = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

    # create a session to run
    with tf.Session() as sess:
        # if use var=tf.Variable, then sess must run init operate. or when sess.run(var) will get not init error.
        sess.run(tf.global_variables_initializer())
        for i in range(201):
            sess.run(train_op)
            if i % 20 == 0:
                # sess.run(w), sess.run(b) same as sess.run([w, b])
                print sess.run(w).T, sess.run(b)


def linear_regression_batch():
    import tensorflow as tf
    import numpy as np

    # prepared  train data
    M = 100  # train data count
    N = 2  # train data feature dimension
    w_data = np.mat([[1.0, 3.0]]).T
    b_data = 10
    x_data = np.random.randn(M, N).astype(np.float32)  # randn return float64 number
    y_data = np.mat(x_data) * w_data + 10 + np.random.randn(M, 1) * 0.33  # last term is random error term

    # define model graph and loss function
    batch_size = 1
    # use tf tensor type var just like use np.array
    X = tf.placeholder("float", [batch_size, N])  # declare a graph node, but not init it immediately.
    Y = tf.placeholder("float", [batch_size, 1])
    w = tf.Variable(tf.random_uniform([N, 1], -1, 1))
    b = tf.Variable(tf.random_uniform([1], -1, 1))
    loss = tf.reduce_mean(tf.square(Y - tf.matmul(X, w) - b))

    # choose optimizer and operator
    train_op = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

    # run model use session
    with tf.Session() as sess:
        # init all global var in graph
        sess.run(tf.global_variables_initializer())

        for epoch in range(200 * batch_size / M):
            i = 0
            while i < M:
                sess.run(train_op, feed_dict={X: x_data[i: i + batch_size], Y: y_data[i: i + batch_size]})
                i += batch_size
            print "epoch: {}, w: {}, b: {}".format(epoch, sess.run(w).T, sess.run(b))


if __name__ == '__main__':
    linear_regression()
    # linear_regression_batch()
