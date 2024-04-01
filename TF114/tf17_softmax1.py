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
hypothesis = tf.compat.v1.tf.matmul(x, w) + b



loss = tf.reduce_mean(tf.compat.v1.square(hypothesis - y))

# optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=1e-5)
# train = optimizer.minimize(loss)
#combine 
train = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=1e-5).minimize(loss)

# Initialize global variables
init = tf.compat.v1.global_variables_initializer()

# Start a new TF session
with tf.compat.v1.Session() as sess:
    # Run the initializer
    sess.run(init)

    # Training cycle
    for step in range(2001):
        _, cost_val = sess.run([train, loss], feed_dict={x: x_data, y: y_data})

        if step % 200 == 0:
            print("Step: ", step, ", Cost: ", cost_val)

    print('Training Finished!')

    # Test model on training set
    prediction = tf.argmax(hypothesis, 1)
    correct_prediction = tf.equal(prediction, tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    print("Accuracy: ", sess.run(accuracy, feed_dict={x: x_data, y: y_data}))


