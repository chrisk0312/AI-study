import tensorflow as tf
tf.set_random_seed(777)  

#1 data
x = [1,2,3,4,5]
y = [3,5,7,9,11]

# w = tf.Variable(111, dtype=tf.float32)
# b = tf.Variable(0, dtype=tf.float32)
w = tf.Variable(tf.random_normal([1]), dtype=tf.float32)
b = tf.Variable(tf.random_normal([1]), dtype=tf.float32)

sess = tf.compat.v1.Session()
sess.run(tf.global_variables_initializer())
print(sess.run(w)) # [2.2086694]

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
        sess.run(train)
        if step % 20 == 0:
            print(step, sess.run(loss), sess.run(w), sess.run(b))

# 0 1.1094682 [2.2719872] [-0.79653]
# 20 0.4804348 [2.4490511] [-0.62489617]
# 40 0.41955757 [2.4205227] [-0.5182366]
# 60 0.36640272 [2.392986] [-0.4188051]
# 80 0.3199821 [2.3672493] [-0.32588643]