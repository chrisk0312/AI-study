import tensorflow as tf
import numpy as np
tf.set_random_seed(777)

x_data = [[1, 2, 1, 1],[2,1,3,2],[3,1,3,4],[4,1,5,5],[1,7,5,5],[1,2,5,6],[1,6,6,6],[1,7,7,7]]
y_data = [[0,0,1],[0,0,1],[0,0,1],[0,1,0],[0,1,0],[0,1,0],[1,0,0],[1,0,0]]

x = tf.compat.v1.placeholder(tf.float32, shape=[None, 4])
w = tf.compat.v1.Variable(tf.random_normal([4, 3]), name='weight')
b = tf.compat.v1.Variable(tf.zeros([1,3]), name='bias')
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 3])

#model
hypothesis = tf.nn.softmax(tf.compat.v1.matmul(x, w) + b)

# loss = tf.reduce_mean(tf.compat.v1.square(hypothesis - y))
loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(hypothesis), axis=1))

# optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=1e-5)
# train = optimizer.minimize(loss)
#combine 
train = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=1e-1).minimize(loss)

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

epochs = 20001
for step in range(epochs):
    _, cost_val, w_val, b_val = sess.run([train, loss, w, b], feed_dict={x: x_data, y: y_data})
    
    if step%20 == 0:
        print(step,'loss :', cost_val)
print(w_val, b_val)        

y_pred = sess.run(hypothesis, feed_dict={x: x_data})
print(y_pred)
y_pred = sess.run(tf.argmax(y_pred, 1))
print(y_pred)
y_data = np.argmax(y_data, axis=1)
print(y_data)

from sklearn.metrics import accuracy_score
acc = accuracy_score(y_pred, y_data)
print('acc: ', acc)

