import tensorflow as tf
import numpy as np
r = np.random.randint(1,1000)
r = 147
tf.compat.v1.set_random_seed(r)

from sklearn.datasets import fetch_california_housing

x, y = fetch_california_housing(return_X_y=True)

import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder
# y = OneHotEncoder(sparse=False).fit_transform(y.reshape(-1,1))
x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.8,random_state=r)
scaler = StandardScaler().fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

print(x_train.shape,y_train.shape,x_test.shape,y_test.shape)
# (16512, 8) (16512, 6) (4128, 8) (4128, 6)
print(np.unique(y_train,return_counts=True))
# (array([0.14999, 0.175  , 0.225  , ..., 4.991  , 5.     , 5.00001]), array([  3,   1,   3, ...,   1,  24, 776], dtype=int64))

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

IS_TRAIN = True
layer1 = Dense_layer(128,x_train.shape[1],tf.nn.relu).get_layer(x)
layer2 = Dense_layer(128,128,tf.nn.relu).get_layer(layer1)
if IS_TRAIN:
    layer2 = tf.compat.v1.nn.dropout(layer2,keep_prob=0.9)
layer3 = Dense_layer(64,128,tf.nn.sigmoid).get_layer(layer2)
layer4 = Dense_layer(64,64,tf.nn.relu).get_layer(layer3)
layer5 = Dense_layer(32,64,tf.nn.relu).get_layer(layer4)
layer6 = Dense_layer(16,32,tf.nn.sigmoid).get_layer(layer5)
hypothesis = Dense_layer(1,16).get_layer(layer6)

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
        
    IS_TRAIN = False
    pred = sess.run(hypothesis,feed_dict={x:x_test})
    pred = np.around(pred)
    
print("pred: ",pred.shape)
from sklearn.metrics import r2_score
r2 = r2_score(pred,y_test)

print("R2: ",r2)
print("Random state: ",r)

# pred:  (4128, 1)
# R2:  -5525.268318677757
# Random state:  147