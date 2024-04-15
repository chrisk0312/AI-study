import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
tf.compat.v1.set_random_seed(47)
import warnings
warnings.filterwarnings('ignore')
tf.compat.v1.disable_eager_execution()  # 이래야 tf1 코드를 쓸 수 있음
# tf.compat.v1.enable_eager_execution()
print(tf.__version__)

#1 data
from tensorflow.keras.datasets import cifar10
(x_train,y_train), (x_test,y_test) = cifar10.load_data()
print(x_train.shape,y_train.shape,x_test.shape,y_test.shape)
# (50000, 32, 32, 3) (50000, 1) (10000, 32, 32, 3) (10000, 1)
print(np.unique(y_train,return_counts=True))
# (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), array([5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000],
from tensorflow.keras.utils import to_categorical
x_train = x_train / 255.
x_test = x_test / 255.
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
print(x_train.shape,y_train.shape,x_test.shape,y_test.shape)
# (50000, 32, 32, 3) (50000, 10) (10000, 32, 32, 3) (10000, 10)

#2 model
x = tf.compat.v1.placeholder(tf.float32,shape=[None,32,32,3])
y = tf.compat.v1.placeholder(tf.float32,shape=[None,10])

w1 = tf.compat.v1.get_variable('w1',shape=[2,2,3,128])# [kernal_width,kernal_height,channel,filter]
b1 = tf.compat.v1.Variable(tf.zeros([128]),name='b1')
c1 = tf.nn.relu(tf.nn.conv2d(x,w1,strides=[1,1,1,1],padding='VALID',name='conv2d_1')+b1) # stride=(2,2)는 이렇게 표기한다 stride=[1,2,2,1]
print("c1",c1)

dr1 = tf.nn.dropout(c1,rate=0.01)
m1 = tf.nn.max_pool2d(dr1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')
print("m1",m1)

w2 = tf.compat.v1.get_variable('w2',shape=[3,3,128,64])
b2 = tf.compat.v1.Variable(tf.zeros([64]),name='b2')
c2 = tf.nn.selu(tf.nn.conv2d(m1,w2,strides=[1,1,1,1],padding='SAME',name='conv2d_2')+b2)
print("c2",c2)

dr2 = tf.nn.dropout(c2,rate=0.01)
m2 = tf.nn.max_pool2d(dr2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')
print("m2",m2)

w3 = tf.compat.v1.get_variable('w3',shape=[3,3,64,32])
b3 = tf.compat.v1.Variable(tf.zeros([32]),name='b3')
c3 = tf.nn.elu(tf.nn.conv2d(m2,w3,strides=[1,1,1,1],padding='SAME',name='conv2d_2')+b3)
print("c2",c3)

flatten = tf.reshape(c3,shape=[-1,c3.shape[1]*c3.shape[2]*c3.shape[3]])
print('flatten',flatten)

w4 = tf.compat.v1.get_variable('w4',shape=[flatten.shape[1],100])
b4 = tf.compat.v1.Variable(tf.zeros([100]),name='b4')
d1 = tf.nn.relu(tf.matmul(flatten,w4) + b4)
print("d1",d1)

dr3 = tf.nn.dropout(d1,rate=0.01)

w5 = tf.compat.v1.get_variable('w5',shape=[100,10])
b5 = tf.compat.v1.Variable(tf.zeros([10]),name='b5')
hypothesis = tf.nn.softmax(tf.matmul(dr3,w5) + b5)
print("hypothesis",hypothesis)

loss_fn = tf.reduce_sum(tf.square(hypothesis-y))
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.001)
train = optimizer.minimize(loss_fn)

#3 compile & fit
import time
EPOCHS = 40
BATCH = 1000
with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
    
    # batch
    x_train_list = []
    y_train_list = []
    for i in range((x_train.shape[0]//BATCH) +1):
        s = i*BATCH
        e = (i+1)*BATCH
        if e > x_train.shape[0]:
            x_train_list.append(x_train[s:])
            y_train_list.append(y_train[s:])
        else:
            x_train_list.append(x_train[s:e])
            y_train_list.append(y_train[s:e])
    print("x_list, y_list length:",len(x_train_list),len(y_train_list))
    for x_data, y_data in zip(x_train_list,y_train_list):
        print("split by train x_data, y_data")
        print(len(x_data),len(y_data))
        
    x_test_list = []
    y_test_list = []
    for i in range((x_test.shape[0]//BATCH) +1):
        s = i*BATCH
        e = (i+1)*BATCH
        if e > x_test.shape[0]:
            x_test_list.append(x_test[s:])
            y_test_list.append(y_test[s:])
        else:
            x_test_list.append(x_test[s:e])
            y_test_list.append(y_test[s:e])
    print("x_list, y_list length:",len(x_test_list),len(y_test_list))
    for x_data, y_data in zip(x_test_list,y_test_list):
        print("split by test x_data, y_data")
        print(len(x_data),len(y_data))
    
    # Early Stopping
    best_loss = 987654321
    best_acc = 0
    patient_stacking = 0    
    PAITIENT = 100
    
    for step in range(1,EPOCHS+1):
        if patient_stacking >= PAITIENT:
            print(f"Early Stop: {step}epos")
            break
        patient_stacking += 1
        
        loss = 0
        for i, (x_data ,y_data) in enumerate(zip(x_train_list,y_train_list)):
            _, _loss = sess.run([train,loss_fn], feed_dict={x:x_data,y:y_data})
            loss += _loss / len(x_train_list)
            print('.',end='')
        else:
            print('\n1epo sucess')
        
        arg_pred = None
        for i, (x_data ,y_data) in enumerate(zip(x_test_list,y_test_list)):
            if i == 0:
                arg_pred = np.argmax(sess.run(hypothesis,feed_dict={x:x_data}),axis=1)
                continue
            arg_pred = np.concatenate([arg_pred,np.argmax(sess.run(hypothesis,feed_dict={x:x_data}),axis=1)],axis=0)
        arg_y = np.argmax(y_test,axis=1)
        _acc = accuracy_score(arg_y,arg_pred)
            
        if step%1 == 0:
            print(f"{step} epo loss={loss:.10f} acc={_acc:.6f}")
            
    IS_TRAIN = False
    pred = None
    for i, (x_data ,y_data) in enumerate(zip(x_test_list,y_test_list)):
        if i == 0:
            pred = sess.run([hypothesis], feed_dict={x:x_data,y:y_data})
            continue
        pred = np.concatenate([pred,sess.run([hypothesis], feed_dict={x:x_data,y:y_data})],axis=1)
    pred = np.array(pred)
    print("pred: ",pred.shape, " before argmax")
    pred = pred[0]

    pred = np.argmax(pred,axis=1)
    y_test = np.argmax(y_test,axis=1)
    
print("pred: ",pred.shape)
acc = accuracy_score(pred,y_test)

print("ACC: ",acc)
print("Random state: ",47)

# ACC:  0.7024
# Random state:  47
