import tensorflow as tf
import numpy as np
r = 8603
tf.compat.v1.set_random_seed(r)

# data
x_data = [
    [0,0],
    [0,1],
    [1,0],
    [1,1]
    ]

y_data = [[0],[1],[1],[0]]

x = tf.compat.v1.placeholder(tf.float32,shape=[None,2])
y = tf.compat.v1.placeholder(tf.float32,shape=[None,1])

# model.add(Dense(10,input_dim=2))
w1 = tf.compat.v1.Variable(tf.random_normal([2,10]))
b1 = tf.compat.v1.Variable(tf.zeros([1,10]),dtype=tf.float32)
layer1 = tf.nn.relu(tf.compat.v1.matmul(x,w1) + b1)

w2 = tf.compat.v1.Variable(tf.random_normal([10,5]))
b2 = tf.compat.v1.Variable(tf.zeros([1,5]),dtype=tf.float32)
layer2 = tf.nn.relu(tf.compat.v1.matmul(layer1,w2) + b2)

w3 = tf.compat.v1.Variable(tf.random_normal([5,5]))
b3 = tf.compat.v1.Variable(tf.zeros([1,5]),dtype=tf.float32)
layer3 = tf.nn.relu(tf.compat.v1.matmul(layer2,w3) + b3)

w4 = tf.compat.v1.Variable(tf.random_normal([5,5]))
b4 = tf.compat.v1.Variable(tf.zeros([1,5]),dtype=tf.float32)
layer4 = tf.nn.relu(tf.compat.v1.matmul(layer3,w4) + b4)

w5 = tf.compat.v1.Variable(tf.random_normal([5,3]))
b5 = tf.compat.v1.Variable(tf.zeros([1,3]),dtype=tf.float32)
layer5 = tf.nn.relu(tf.compat.v1.matmul(layer4,w5) + b5)

w = tf.compat.v1.Variable(tf.random_normal([3,1]))
b = tf.compat.v1.Variable(tf.zeros([1,1]),dtype=tf.float32)
hypothesis = tf.sigmoid(tf.add(tf.matmul(layer5,w),b))

loss_fn = -tf.reduce_mean(y*tf.log(hypothesis) + (1-y)*tf.log(1-hypothesis))
optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0001)

train = optimizer.minimize(loss_fn)

EPOCHS = 5000
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(1,EPOCHS+1):
        _, loss = sess.run([train,loss_fn],feed_dict={x:x_data,y:y_data})
        print(f"{step}epo loss={loss}")
        
    pred = sess.run(hypothesis,feed_dict={x:x_data})
    pred = np.around(pred)
    
print("y ture: ",y_data)
print("pred: ",pred)
from sklearn.metrics import accuracy_score
acc = accuracy_score(pred,y_data)

print("ACC: ",acc)
