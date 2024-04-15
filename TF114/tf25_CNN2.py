import tensorflow as tf
import numpy as np
tf.compat.v1.set_random_seed(47)

# data
x_train = np.array([[[[1],[2],[3]],
                     [[4],[5],[6]],
                     [[7],[8],[9]]]])
print(x_train.shape) # (1, 3, 3, 1)

x = tf.compat.v1.placeholder(tf.float32,shape=[None,3,3,1])
w = tf.compat.v1.constant([[[[1.]],[[0.]]],
                           [[[1.]],[[0.]]]])
print(w)

c1 = tf.nn.conv2d(x,w,strides=[1,1,1,1],padding='VALID')
print(c1)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(c1,feed_dict={x:x_train}))