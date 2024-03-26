# change lr and  make epoch less than 101
# step = less than 100, w=1.99, b=0.99

import tensorflow as tf
tf.set_random_seed(777)  

#1 data
x_data = [1,2,3,4,5]
y_data = [3,5,7,9,11]

x = tf.placeholder(tf.float32, shape=[None])
y = tf.placeholder(tf.float32, shape=[None])

w = tf.Variable(tf.random_normal([1]), dtype=tf.float32)
b = tf.Variable(tf.random_normal([1]), dtype=tf.float32)


#2 model
hypothsis = x * w + b

#compile, train
loss = tf.reduce_mean(tf.square(hypothsis - y)) # mse

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(loss)


with tf.compat.v1.Session() as sess:
    sess.run(tf.global_variables_initializer())

    epochs = 100
    for step in range(epochs):
        _, loss_val, w_val, b_val = sess.run([train, loss, w, b], feed_dict={x: x_data, y: y_data})
        if step % 20 == 0:
            print(step, loss_val, w_val, b_val)
    print("=====================================")
    #predict
    x_pred_data = [6,7,8]
    # x_test = tf.placeholder(tf.float32, shape=[None])

    # #1. python code
    # y_pred = x_pred_data * w_val + b_val
    # print("6,7,8 예측: ", y_pred)

    #2. insert placeholder
    y_pred2 = x_pred_data * w_val + b_val
    print("6,7,8 예측: ", y_pred2)

# 6,7,8 예측:  [13.0919587  15.13044041 17.16892213]
# 6,7,8 예측:  [13.091959 15.130441 17.168922]
