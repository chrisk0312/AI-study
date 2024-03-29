import tensorflow as tf
tf.compat.v1.set_random_seed(777)

x_data = [[1, 2], [2, 3], [3, 1], [4, 3], [5, 3], [6, 2]] # (6, 2)
y_data = [[0], [0], [0], [1], [1], [1]] # (6, 1)

# Placeholders for the input data
x = tf.placeholder(tf.float32, shape=[None, 2])
y = tf.placeholder(tf.float32, shape=[None, 1])

# Variables for the weights and bias
w = tf.compat.v1.Variable(tf.compat.v1.random_normal([2, 1]), dtype=tf.float32)
b = tf.compat.v1.Variable(tf.compat.v1.zeros([1]), dtype=tf.float32)

# Hypothesis using sigmoid: tf.div(1., 1. + tf.exp(tf.matmul(X, W)))
hypothesis = tf.compat.v1.matmul(x, w) + b

# Loss function (cross entropy)
loss = -tf.reduce_mean(tf.compat.v1.square(hypothesis - y))

# Optimizer (gradient descent)
optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(loss)

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

epochs  = 101
for step in range(epochs):
    _, cost_val, w_val, b_val = sess.run([train, loss,w,b], feed_dict={x:x_data, y:y_data})
    if step % 20 == 0:
        print(step, "loss :", cost_val)

print(w_val, b_val)

print(type(w_val))

#evaluate
x_test =  tf.compat.v1.placeholder(tf.float32, shape=[None, 2])
y_pred = tf.matmul(x_test, w_val) + b_val
y_predict = sess.run(y_pred, feed_dict={x_test:x_data})
print(y_predict)

from sklearn.metrics import r2_score, mean_squared_error
r2 = r2_score(y_data, y_predict)
print("R2 score: ", r2) #R2 score:  -69.87945269027755

mse = mean_squared_error(y_data, y_predict)
print("MSE: ", mse) #MSE:  17.719863172569386

sess.close()
        