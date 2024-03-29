from sklearn.datasets import load_breast_cancer
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


datasets = load_breast_cancer()
x,y = datasets.data, datasets.target

x = x[ y !=2]
y = y[ y !=2]

print(y, y.shape)

# Reshape y to 2D array
y = y.reshape(-1, 1)

# Split data into training and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Standardize the data
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Placeholders for the input data
X = tf.placeholder(tf.float32, shape=[None, x_train.shape[1]])
Y = tf.placeholder(tf.float32, shape=[None, 1])

# Variables for the weights and bias
W = tf.Variable(tf.random_normal([x_train.shape[1], 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

# Hypothesis using sigmoid
hypothesis = tf.sigmoid(tf.matmul(X, W) + b)

# Loss function (cross entropy)
loss = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis))

# Optimizer (gradient descent)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(loss)

# Accuracy computation
predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))

# Session
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # Training
    for step in range(10001):
        loss_val, _ = sess.run([loss, train], feed_dict={X: x_train, Y: y_train})
        if step % 200 == 0:
            print(step, loss_val)

    # Accuracy report
    h, c, a = sess.run([hypothesis, predicted, accuracy], feed_dict={X: x_test, Y: y_test})
    print("\nHypothesis: ", h, "\nCorrect (Y): ", c, "\nAccuracy: ", a)