import tensorflow as tf
from tensorflow.keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
import numpy as np

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print(x_train.shape, y_train.shape, x_test.shape, y_test.shape) # (60000, 28, 28) (60000, 10) (10000, 28, 28) (10000, 10)

x_train = x_train.reshape(60000,28*28).astype('float32')/255. 
x_test = x_test.reshape(10000,28*28).astype('float32')/255.

print(x_train.shape, y_train.shape, x_test.shape, y_test.shape) # (60000, 784) (60000, 10) (10000, 784) (10000, 10)

x = tf.compat.v1.placeholder(tf.float32,shape=[None,784])
y = tf.compat.v1.placeholder(tf.float32,shape=[None,10])


w1 = tf.compat.v1.Variable(tf.random.normal([784, 128]), name='w1')
b1 = tf.compat.v1.Variable(tf.zeros([128]), name='b1')
layer1 = tf.compat.v1.matmul(x, w1) + b1
layer1 = tf.compat.v1.nn.dropout(layer1, rate=0.3)

w2 = tf.compat.v1.Variable(tf.random.normal([128, 64]), name='w2')
b2 = tf.compat.v1.Variable(tf.zeros([64]), name='b2')
layer2 = tf.compat.v1.matmul(layer1, w2) + b2
layer2 = tf.compat.v1.nn.relu(layer2)
layer2 = tf.compat.v1.nn.dropout(layer2, rate=0.3)

w3 = tf.compat.v1.Variable(tf.random.normal([64, 32]), name='w3')
b3 = tf.compat.v1.Variable(tf.zeros([32]),name='b3')
layer3 = tf.compat.v1.matmul(layer2,w3) + b3
layer3 = tf.compat.v1.nn.relu(layer3)
layer3 = tf.compat.v1.nn.dropout(layer3,rate=0.3)

w4 = tf.compat.v1.Variable(tf.random.normal([32, 10]), name='w4')
b4 = tf.compat.v1.Variable(tf.zeros([10]),name='b4')
layer4 = tf.compat.v1.matmul(layer3,w4) + b4
hypothesis = tf.compat.v1.nn.softmax(layer4)

# w1 = tf.compat.v1.Variable('w1', shape=[784,128])
# b1 = tf.compat.v1.Variable(tf.zeros([128]),name='b1')
# layer1 = tf.compat.v1.matmul(x,w1) + b1
# layer1 = tf.compat.v1.nn.dropout(layer1,rate=0.3)


# w2 = tf.compat.v1.Variable('w2', shape=[128,64])
# b2 = tf.compat.v1.Variable(tf.zeros([64]),name='b2')
# layer2 = tf.compat.v1.matmul(layer1,w2) + b2
# layer2 = tf.compat.v1.nn.relu(layer2)
# layer2 = tf.compat.v1.nn.dropout(layer2,rate=0.3)

# w3 = tf.compat.v1.Variable('w3', shape=[64,32])
# b3 = tf.compat.v1.Variable(tf.zeros([32]),name='b3')
# layer3 = tf.compat.v1.matmul(layer2,w3) + b3
# layer3 = tf.compat.v1.nn.relu(layer3)
# layer3 = tf.compat.v1.nn.dropout(layer3,rate=0.3)

# w4 = tf.compat.v1.Variable('w4', shape=[32,10])
# b4 = tf.compat.v1.Variable(tf.zeros([10]),name='b4')
# layer4 = tf.compat.v1.matmul(layer3,w4) + b4
# hypothesis = tf.compat.v1.nn.softmax(layer4)

loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(tf.nn.softmax(hypothesis)), axis=1))
# loss = tf.reduce_mean(-tf.reduce_sum(y*tf.compat.v1.log_softmax(hypothesis),axis=1))


train = tf.compat.v1.train.AdamOptimizer(learning_rate=1e-3).minimize(loss)

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

batch_size = 1000
training_epochs = 100
total_batch = int(len(x_train)/batch_size)
print(total_batch) # 60

for step in range(training_epochs):
    avg_cost = 0
    for i in range(total_batch):
        start = i * batch_size
        end = start + batch_size
        batch_x, batch_y = x_train[start:end], y_train[start:end]
        feed_dict = {x:batch_x,y:batch_y}
        cost_val, _, w_val, b_val = sess.run([loss,train, w4,b4],feed_dict=feed_dict)
        
        avg_cost += cost_val/total_batch
    
    if step%20 == 0:
        print(f"{step}epo loss={cost_val}")
        

