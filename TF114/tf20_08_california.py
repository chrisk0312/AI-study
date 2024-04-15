from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import accuracy_score, r2_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing

import warnings
warnings.filterwarnings(action='ignore')
#data

x, y = fetch_california_housing(return_X_y=True)

# model 
import warnings
warnings.filterwarnings('ignore')

x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=333, train_size=0.8,
)

print(x_train.shape,y_train.shape)
print(np.unique(y_train,return_counts=True))


import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
scaler = StandardScaler().fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

print(x_train.shape,y_train.shape,x_test.shape,y_test.shape)
# (77034, 13) (77034,) (19259, 13) (19259,)

x = tf.compat.v1.placeholder(tf.float32,shape=[None,x_train.shape[1]])
y = tf.compat.v1.placeholder(tf.float32,shape=[None,])

class Dense_layer():
    def __init__(self, output_dim, input_dim, activation=None) -> None:
        self.w = tf.compat.v1.Variable(tf.random_normal([input_dim,output_dim]))
        self.b = tf.compat.v1.Variable(tf.zeros([1,output_dim]),dtype=tf.float32)
        self.activation = activation
    def get_layer(self,x):
        result = tf.matmul(x,self.w) + self.b
        if self.activation is not None:
            return self.activation(result)
        return result


layer1 = Dense_layer(128,x_train.shape[1],tf.nn.relu).get_layer(x)
layer2 = Dense_layer(128,128,tf.nn.relu).get_layer(layer1)
layer3 = Dense_layer(64,128,tf.nn.sigmoid).get_layer(layer2)
layer4 = Dense_layer(64,64,tf.nn.relu).get_layer(layer3)
layer5 = Dense_layer(32,64,tf.nn.relu).get_layer(layer4)
layer6 = Dense_layer(32,32,tf.nn.sigmoid).get_layer(layer5)
hypothesis = Dense_layer(1,32).get_layer(layer6)

loss_fn = tf.reduce_mean(tf.square(hypothesis-y))
optimizer = tf.train.AdamOptimizer(learning_rate=0.001)

train = optimizer.minimize(loss_fn)

EPOCHS = 1000
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(1,EPOCHS+1):
        _, loss = sess.run([train,loss_fn],feed_dict={x:x_train,y:y_train})
        if step%100 == 0:
            print(f"{step}epo loss={loss}")
        
    pred = sess.run(hypothesis,feed_dict={x:x_test})
    pred = np.around(pred)
    
print("pred: ",pred)
r2 = r2_score(y_test,pred)

print("R2: ",r2)
print("Random state: ",333)

# R2:  -0.15725737514682137
# Random state:  333