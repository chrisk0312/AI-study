x1_data = [73., 93., 89., 96., 73.]
x2_data = [80., 88., 91., 98., 66.]
x3_data = [75., 93., 90., 100., 70.]
y_data = [152., 185., 180., 196., 142.]

import tensorflow as tf
tf.compat.v1.set_random_seed(777)

# Placeholders for the input data
x1 = tf.compat.v1.placeholder(tf.float32)
x2 = tf.compat.v1.placeholder(tf.float32, shape=[None])
x3 = tf.compat.v1.placeholder(tf.float32, shape=[None])
y = tf.compat.v1.placeholder(tf.float32)

# Variables for the weights
w1 = tf.compat.v1.Variable(tf.compat.v1.random_normal([1]))
w2 = tf.compat.v1.Variable(tf.compat.v1.random_normal([1]), dtype= tf.float32)
w3 = tf.compat.v1.Variable(tf.compat.v1.random_normal([1]), dtype = tf.float32)
b = tf.compat.v1.Variable(tf.compat.v1.random_normal([1]))

# Hypothesis (linear model)
hypothesis = x1 * w1 + x2 * w2 + x3 * w3 + b

# Loss function (mean squared error)
loss = tf.reduce_mean(tf.compat.v1.square(hypothesis - y))

# Optimizer (gradient descent)
optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(loss)

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

epochs = 2000
for step in range(epochs):
    cost_val,_ =  sess.run([loss, train], feed_dict={x1: x1_data, x2: x2_data, x3: x3_data, y: y_data})
    if step % 20 == 0:
        print(step, "loss: ", cost_val)
sess.close()

