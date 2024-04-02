import tensorflow as tf
tf.compat.v1.set_random_seed(777)

x_data = [[0.0],[0,1],[1,0],[1,1]]
y_data = [[0],[1],[1],[0]]

x = tf.compat.v1.placeholder(tf.float32, shape=[None, 2])
y = tf.compat.v1.placeholder(tf.float32, shape = [None, 1])
w = tf.compat.v1.Variable(tf.random_normal([2, 1]), name='weight')
b = tf.compat.v1.Variable(tf.zeros([1]), name='bias')

# Model
hypothesis = tf.sigmoid(tf.matmul(x, w) + b)

# Cost/Loss function
cost = -tf.reduce_mean(y * tf.log(hypothesis) + (1 - y) * tf.log(1 - hypothesis))

# Optimizer
train = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

# Accuracy computation
# True if hypothesis>0.5 else False
predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, y), dtype=tf.float32))

# Initialize global variables
init = tf.compat.v1.global_variables_initializer()

# Start a new TF session
with tf.compat.v1.Session() as sess:
    # Run the initializer
    sess.run(init)

    # Training cycle
    for step in range(10001):
        # Add a zero to the first element of x_data
        x_data_feed = [[0.0, 0.0]] + x_data[1:]
        _, cost_val = sess.run([train, cost], feed_dict={x: x_data_feed, y: y_data})

        if step % 1000 == 0:
            print("Step: ", step, ", Cost: ", cost_val)

    print('Training Finished!')

    # Accuracy
    h, p, a = sess.run([hypothesis, predicted, accuracy], feed_dict={x: x_data_feed, y: y_data})
    print("\nHypothesis: ", h, "\nPredicted: ", p, "\nAccuracy: ", a)