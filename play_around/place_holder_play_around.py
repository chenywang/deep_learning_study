# View more python learning tutorial on my Youtube and Youku channel!!!

# Youtube video tutorial: https://www.youtube.com/channel/UCdyjiB5H8Pu7aDTNVXTTpcg
# Youku video tutorial: http://i.youku.com/pythontutorial

"""
Please note, this code is only for python 3+. If you are using python 2+, please modify the code accordingly.
"""
import tensorflow as tf

input1 = tf.placeholder(tf.float32, [2, ])
input2 = tf.placeholder(tf.float32)
output = tf.multiply(input1, input2)
print "input1", type(input1)
print "output", type(output)
print "shape of [7., 2.] is [2, ]"

with tf.Session() as sess:
    print sess.run(output, feed_dict={input1: [7., 2.], input2: [2., 3.]})
