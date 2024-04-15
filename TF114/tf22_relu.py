import tensorflow as tf
import numpy as np
r = np.random.randint(1,1000)
# r = 51
tf.compat.v1.set_random_seed(r)

from sklearn.datasets import load_breast_cancer

x_data, y_data = load_breast_cancer(return_X_y=True)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

x_train, x_test, y_train, y_test = train_test_split(x_data,y_data,train_size=0.8,random_state=r)
scaler = StandardScaler().fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

print(x_train.shape,y_train.shape,x_test.shape,y_test.shape)
# (455, 30) (455,) (114, 30) (114,)

x = tf.compat.v1.placeholder(tf.float32,shape=[None,x_train.shape[1]])
y = tf.compat.v1.placeholder(tf.float32,shape=[None,])


# model
IS_TRAIN = True

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
if IS_TRAIN:
    layer2 = tf.compat.v1.nn.dropout(layer2,keep_prob=0.5)
layer3 = Dense_layer(128,256,tf.nn.relu).get_layer(layer2)
layer4 = Dense_layer(64,128,tf.nn.relu).get_layer(layer3)
layer5 = Dense_layer(32,64,tf.nn.sigmoid).get_layer(layer4)
hypothesis = Dense_layer(1,32,tf.nn.sigmoid).get_layer(layer5)

loss_fn = -tf.reduce_mean(y*tf.log(hypothesis) + (1-y)*tf.log(1-hypothesis))
optimizer = tf.train.AdamOptimizer(learning_rate=0.001)

train = optimizer.minimize(loss_fn)

EPOCHS = 5000
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(1,EPOCHS+1):
        _, loss = sess.run([train,loss_fn],feed_dict={x:x_train,y:y_train})
        if step%200 == 0:
            print(f"{step}epo loss={loss}")
        
    IS_TRAIN = False
    pred = sess.run(hypothesis,feed_dict={x:x_test})
    pred = np.around(pred)
    
print("pred: ",pred.shape)
from sklearn.metrics import accuracy_score
acc = accuracy_score(pred,y_test)

print("ACC: ",acc)
print("Random state: ",r)

# origin
# ACC:  0.6666666666666666
# Random state:  51

# dropout
# ACC:  0.6666666666666666
# Random state:  51

# ACC:  0.6666666666666666
# Random state:  51

# pred:  (114, 1)
# ACC:  0.5964912280701754
# Random state:  840