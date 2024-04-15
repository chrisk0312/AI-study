import tensorflow as tf
import numpy as np
r = 8603
tf.compat.v1.set_random_seed(r)

# data
x_data = [
    [0,0],
    [0,1],
    [1,0],
    [1,1]
    ]

y_data = [[0],[1],[1],[0]]

x = tf.compat.v1.placeholder(tf.float32,shape=[None,2])
y = tf.compat.v1.placeholder(tf.float32,shape=[None,1])

class Dense_layer():
    def __init__(self, output_dim, input_dim) -> None:
        self.w = tf.compat.v1.Variable(tf.random_normal([input_dim,output_dim]))
        self.b = tf.compat.v1.Variable(tf.zeros([1,output_dim]),dtype=tf.float32)
    def get_layer(self,x):
        return tf.matmul(x,self.w) + self.b

# model.add(Dense(10,input_dim=2))
layer1 = Dense_layer(10,2).get_layer(x)
relu1 = tf.nn.relu(layer1)
layer2 = Dense_layer(5,10).get_layer(relu1)
relu2 = tf.nn.relu(layer2)
layer3 = Dense_layer(5,5).get_layer(relu2)
relu3 = tf.nn.relu(layer3)
layer4 = Dense_layer(5,5).get_layer(relu3)
relu4 = tf.nn.relu(layer4)
layer5 = Dense_layer(3,5).get_layer(relu4)
relu5 = tf.nn.relu(layer5)
layer6 = Dense_layer(1,3).get_layer(relu5)
hypothesis = tf.nn.sigmoid(layer6)

loss_fn = -tf.reduce_mean(y*tf.log(hypothesis) + (1-y)*tf.log(1-hypothesis))
optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0001)

train = optimizer.minimize(loss_fn)

EPOCHS = 5000
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(1,EPOCHS+1):
        _, loss = sess.run([train,loss_fn],feed_dict={x:x_data,y:y_data})
        if step%200 == 0:
            print(f"{step}epo loss={loss}")
        
    pred = sess.run(hypothesis,feed_dict={x:x_data})
    pred = np.around(pred)
    
print("pred: ",pred)
from sklearn.metrics import accuracy_score
acc = accuracy_score(pred,y_data)

print("ACC: ",acc)
