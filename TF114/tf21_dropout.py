import tensorflow as tf
tf.compat.v1.set_random_seed(777)

x_data = [[0.0],[0,1],[1,0],[1,1]]
y_data = [[0],[1],[1],[0]]

#2. model
#layer1 : model.add(Dense(10, input_dim=2))
x = tf.compat.v1.placeholder(tf.float32, shape=[None, 2])
y = tf.compat.v1.placeholder(tf.float32, shape = [None, 1])

w1 = tf.compat.v1.Variable(tf.random_normal([2, 10]), name='weight1')
b1 = tf.compat.v1.Variable(tf.zeros([10]), name='bias')
layer1 = tf.compat.v1.matmul(x, w1) + b1 #(none, 10)

#layer2 : model.add(Dense(9))
w2 = tf.compat.v1.Variable(tf.random_normal([10, 9]), name='weight2')
b2 = tf.compat.v1.Variable(tf.zeros([9]), name='bias')
layer2 = tf.compat.v1.matmul(layer1, w2) + b2
layer2 = tf.compat.v1.nn.dropout(layer2, keep_prob=0.5)


#layer3 : model.add(Dense(8))
w3 = tf.compat.v1.Variable(tf.random_normal([9, 8]), name='weight3')
b3 = tf.compat.v1.Variable(tf.zeros([8]), name='bias') 
layer3 = tf.compat.v1.matmul(layer2, w3) + b3

#layer4 : model.add(Dense(7))
w4 = tf.compat.v1.Variable(tf.random_normal([8, 7]), name='weight4')
b4 = tf.compat.v1.Variable(tf.zeros([7]), name='bias')
layer4 = tf.compat.v1.matmul(layer3, w4) + b4

#output layer : model.add(Dense(1), activation='sigmoid')
w5 = tf.compat.v1.Variable(tf.random_normal([7, 1]), name='weight5')
b5 = tf.compat.v1.Variable(tf.zeros([1]), name='bias')
hypothesis = tf.compat.v1.sigmoid(tf.compat.v1.matmul(layer4, w5) + b5)

# #3-1
# cost = -tf.reduce_mean(y * tf.log(hypothesis) + (1-y) * tf.log(1 - hypothesis))
#     #binary_crossentropy
# train = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

# predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
# accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, y), dtype=tf.float32))

# with tf.Session() as sess :
#     sess.run (tf.global_variables_initializer())
    
#     for step in range(2001):
#         cost_val, _ = sess.run([cost, train], feed_dict={x:x_data, y:y_data})
#         if step % 200 == 0:
#             print(step, cost_val)
            
#     hypo, pred, acc = sess.run([hypothesis, predicted, accuracy], feed_dict={x:x_data, y:y_data})
#     print("예측값 : ", hypo, "\n예측결과 : ", pred, "\n정확도 : ", acc)
# Binary cross entropy cost function
cost = -tf.reduce_mean(y * tf.log(hypothesis) + (1 - y) * tf.log(1 - hypothesis))

# Gradient descent optimizer
train = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

# Accuracy computation
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
        _, cost_val = sess.run([train, cost], feed_dict={x: x_data, y: y_data})

        if step % 1000 == 0:
            print("Step: ", step, ", Cost: ", cost_val)

    print('Training Finished!')

    # Accuracy
    h, p, a = sess.run([hypothesis, predicted, accuracy], feed_dict={x: x_data, y: y_data})
    print("\nHypothesis: ", h, "\nPredicted: ", p, "\nAccuracy: ", a)