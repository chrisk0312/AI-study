import tensorflow as tf
tf.compat.v1.set_random_seed(777)
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

x,y = load_diabetes(return_X_y=True)
print(x.shape, y.shape) #(442, 10) (442,)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle = False, random_state=777)

x = tf.compat.v1.placeholder(tf.float32, shape=[None, 10])
y = tf.compat.v1.placeholder(tf.float32, shape=[None])

w = tf.compat.v1.Variable(tf.random.normal([10, 1]), name='weight')
b = tf.compat.v1.Variable(tf.random.normal([1]), name='bias')

hypothesis = tf.matmul(x, w) + b

loss = tf.reduce_mean(tf.square(hypothesis - y))

optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)
train = optimizer.minimize(loss)

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

epochs  = 101
for step in range(epochs):
    _, cost_val = sess.run([train, loss], feed_dict={x:x_train, y:y_train})
    if step % 20 == 0:
        print(step, cost_val)
        
# Generate predictions for the test set
predictions = sess.run(hypothesis, feed_dict={x: x_test})

# Reshape y_test and predictions to be 1D
y_test = y_test.reshape(-1)
predictions = predictions.reshape(-1)

# Calculate R2 score and MSE
r2 = r2_score(y_test, predictions)
mse = mean_squared_error(y_test, predictions)

print("R2 score: ", r2)
print("MSE: ", mse)