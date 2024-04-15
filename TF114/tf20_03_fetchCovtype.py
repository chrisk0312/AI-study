import tensorflow as tf
import numpy as np
r = np.random.randint(1,1000)
# r = 8603
tf.compat.v1.set_random_seed(r)

from sklearn.datasets import fetch_covtype

x_data, y_data = fetch_covtype(return_X_y=True)
print(np.unique(y_data,return_counts=True))

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder
y_data = OneHotEncoder(sparse=False).fit_transform(y_data.reshape(-1,1))
x_train, x_test, y_train, y_test = train_test_split(x_data,y_data,train_size=0.8,random_state=r)
scaler = StandardScaler().fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

print(x_train.shape,y_train.shape,x_test.shape,y_test.shape)
# (464809, 54) (464809, 7) (116203, 54) (116203, 7)

x = tf.compat.v1.placeholder(tf.float32,shape=[None,x_train.shape[1]])
y = tf.compat.v1.placeholder(tf.float32,shape=[None,y_train.shape[1]])

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


layer1 = Dense_layer(512,x_train.shape[1],tf.nn.relu).get_layer(x)
layer2 = Dense_layer(256,512,tf.nn.relu).get_layer(layer1)
layer3 = Dense_layer(256,256,tf.nn.sigmoid).get_layer(layer2)
layer4 = Dense_layer(128,256,tf.nn.relu).get_layer(layer3)
layer5 = Dense_layer(64,128,tf.nn.relu).get_layer(layer4)
layer6 = Dense_layer(32,64,tf.nn.sigmoid).get_layer(layer5)
hypothesis = Dense_layer(y_train.shape[1],32,tf.nn.softmax).get_layer(layer6)

loss_fn = tf.compat.v1.losses.softmax_cross_entropy(y,hypothesis)
optimizer = tf.train.AdamOptimizer(learning_rate=0.001)

train = optimizer.minimize(loss_fn)

EPOCHS = 100
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(1,EPOCHS+1):
        _, loss = sess.run([train,loss_fn],feed_dict={x:x_train,y:y_train})
        if step%1 == 0:
            print(f"{step}epo loss={loss}")
        
    pred = sess.run(hypothesis,feed_dict={x:x_test})
    pred = np.around(pred)
    
print("pred: ",pred)
from sklearn.metrics import accuracy_score
acc = accuracy_score(pred,y_test)

print("ACC: ",acc)
print("Random state: ",r)

# ACC:  0.509788903900932
# Random state:  343