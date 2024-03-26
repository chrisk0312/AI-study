import tensorflow as tf
tf.set_random_seed(777)  

#1 data
x = [1,2,3,4,5]
y = [3,5,7,9,11]

x = tf.placeholder(tf.float32, shape=[None])
y = tf.placeholder(tf.float32, shape=[None])

w = tf.Variable(tf.random_normal([1]), dtype=tf.float32)
b = tf.Variable(tf.random_normal([1]), dtype=tf.float32)

# sess = tf.compat.v1.Session()
# sess.run(tf.global_variables_initializer())
# print(sess.run(w)) # [2.2086694]

#2 model
hypothsis = x * w + b

#compile, train
loss = tf.reduce_mean(tf.square(hypothsis - y)) # mse

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(loss)

# with tf.compat.v1.Session() as sess:
#     sess.run(tf.global_variables_initializer())

#     epochs = 100
#     for step in range(epochs):
#         sess.run(train)
#         if step % 20 == 0:
#             print(step, sess.run(loss), sess.run(w), sess.run(b))

with tf.compat.v1.Session() as sess:
    sess.run(tf.global_variables_initializer())

    epochs = 100
    for step in range(epochs):
        _, loss_val, w_val, b_val = sess.run([train, loss, w, b], feed_dict={x: [1,2,3,4,5], y: [3,5,7,9,11]})
        if step % 20 == 0:
            print(step, loss_val, w_val, b_val)


# 0 19.95235 [1.0985938] [0.5411478]
# 20 0.006446996 [2.0459719] [0.81725544]
# 40 0.0052748797 [2.0469735] [0.83033496]
# 60 0.004606609 [2.0439153] [0.8414514]
# 80 0.0040229596 [2.0410395] [0.8518348]