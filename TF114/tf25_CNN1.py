import tensorflow as tf
import pandas as pd
import numpy as np
tf.compat.v1.set_random_seed(47)

#1 data
from tensorflow.keras.datasets import mnist
(x_train,y_train), (x_test,y_test) = mnist.load_data()
print(x_train.shape,y_train.shape,x_test.shape,y_test.shape)
# (60000, 28, 28) (60000,) (10000, 28, 28) (10000,)
print(np.unique(y_train,return_counts=True))
# (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), array([5923, 6742, 5958, 6131, 5842, 5421, 5918, 6265, 5851, 5949],
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
x_train = x_train.reshape(-1,28,28,1).astype(np.float32) / 255.
x_test = x_test.reshape(-1,28,28,1).astype(np.float32) / 255.
print(x_train.shape,y_train.shape,x_test.shape,y_test.shape)
# (60000, 28, 28, 1) (60000, 10) (10000, 28, 28, 1) (10000, 10)

#2 model
x = tf.placeholder(tf.float32,shape=[None,28,28,1])
y = tf.placeholder(tf.float32,shape=[None,10])

w1 = tf.compat.v1.get_variable('w1',shape=[2,2,1,64])# [kernal_width,kernal_height,channel,filter]
c1 = tf.nn.conv2d(x,w1,strides=[1,1,1,1],padding='VALID',name='conv2d_1') # stride=(2,2)는 이렇게 표기한다 stride=[1,2,2,1]
# model.add(Conv2d(64,(2,2),stride=1,input_shape(28,28,1)))
print("c1",c1)

w2 = tf.compat.v1.get_variable('w2',shape=[3,3,64,32])
c2 = tf.nn.conv2d(c1,w2,strides=[1,1,1,1],padding='SAME',name='conv2d_2')
print("c2",c2)

#3 compile & fit

#4 predict