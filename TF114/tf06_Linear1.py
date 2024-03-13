import tensorflow as tf
tf.set_random_seed(777)  

#1 data
x = [1,2,3]
y = [1,2,3]

w = tf.Variable(111, dtype=tf.float32)
b = tf.Variable(0, dtype=tf.float32)

#2 model
# y = wx + b
# hypothsis = w * x + b
hypothsis = x * w + b

#3 compile, train
loss = tf.reduce_mean(tf.square(hypothsis - y)) # mse

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(loss)
#model.compile(loss='mse', optimizer='sgd')

sess = tf.compat.v1.Session()
sess.run(tf.global_variables_initializer())

epochs = 100
for step in range(epochs):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(loss), sess.run(w), sess.run(b))
sess.close()    

# 0 44682.156 100.73333 -4.4
# 20 616.13007 25.758467 -35.11292
# 40 197.22882 17.902163 -36.406902
# 60 175.841 16.470543 -34.97625
# 80 159.67189 15.682805 -33.35919