import tensorflow as tf
tf.set_random_seed(777)  

#1 data
x = [1,2,3,4,5]
y = [1,2,3,4,5]

w = tf.Variable(111, dtype=tf.float32)
b = tf.Variable(0, dtype=tf.float32)

hypothsis = x * w + b

#compile, train
loss = tf.reduce_mean(tf.square(hypothsis - y)) # mse

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(loss)

sess = tf.compat.v1.Session()
sess.run(tf.global_variables_initializer())

epochs = 100
for step in range(epochs):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(loss), sess.run(w), sess.run(b))
sess.close()

# 0 77623.92 86.8 -6.6
# 20 128.00407 8.65207 -26.257078
# 40 110.40654 7.8233356 -24.62822
# 60 96.41886 7.3749905 -23.01571
# 80 84.20329 6.9574804 -21.508392