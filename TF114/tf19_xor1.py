import tensorflow as tf
import numpy as np
tf.compat.v1.set_random_seed(47)

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

w = tf.compat.v1.Variable(tf.random_normal([2,1]))
b = tf.compat.v1.Variable(tf.zeros([1]),dtype=tf.float32)

hypothesis = tf.sigmoid(tf.add(tf.matmul(x,w),b))

loss_fn = tf.reduce_mean(y*tf.log(hypothesis) + (1-y)*tf.log(1-hypothesis))
optimizer = tf.train.AdamOptimizer(learning_rate=0.001)

train = optimizer.minimize(loss_fn)

EPOCHS = 1000
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(1,EPOCHS+1):
        _, loss = sess.run([train,loss_fn],feed_dict={x:x_data,y:y_data})
        print(f"{step}epo loss={loss}")
        
    pred = sess.run(hypothesis,feed_dict={x:x_data})
    pred = np.around(pred)
    
print("pred: ",pred)
from sklearn.metrics import accuracy_score
acc = accuracy_score(pred,y_data)

print("ACC: ",acc)
