import tensorflow as tf 
tf.compat.v1.set_random_seed(337)

x_data = [[73, 51, 65], [92, 98., 11], [89, 31, 33], [99, 33, 100], [17, 66, 79]]
y_data = [[152], [185], [180], [205], [142]]


# Placeholders for the input data
x = tf.compat.v1.placeholder(tf.float32, shape=[None, 3])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])

# Variables for the weights and bias
w = tf.compat.v1.Variable(tf.compat.v1.random_normal([3, 1]), name='weight')
b = tf.compat.v1.Variable(tf.compat.v1.random_normal([1]), name='bias')

# Hypothesis (linear model)
hypothesis = tf.compat.v1.matmul(x, w) + b

# Loss function (mean squared error)
loss = tf.reduce_mean(tf.square(hypothesis - y))

# Optimizer (gradient descent)
optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(loss)

# Session
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # Training
    for step in range(2001):
        cost_val,_ = sess.run([loss, train], 
                                       feed_dict={x: x_data, y: y_data})
        if step % 20 == 0:
            print(step, "Loss: ", cost_val)