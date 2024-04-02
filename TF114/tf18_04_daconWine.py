from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

path = "C:\\_data\\DACON\\와인품질분류\\"
train_csv = pd.read_csv(path+"train.csv",index_col=0)
test_csv = pd.read_csv(path+"test.csv",index_col=0)
submit_csv = pd.read_csv(path+"sample_submission.csv")

# print(train_csv.isna().sum(),test_csv.isna().sum()) 결측치 존재안함

# print(train_csv,test_csv,sep='\n') #[5497 rows x 13 columns], [1000 rows x 12 columns]
x = train_csv.drop(['quality'],axis=1)
y = train_csv['quality']

# print(x.shape,y.shape)  #(5497, 12) (5497,)
print(np.unique(y,return_counts=True))
# print(y.shape)          #(5497, 7)

x.loc[x['type'] == 'red', 'type'] = 1
x.loc[x['type'] == 'white', 'type'] = 0
# print(x)
test_csv.loc[test_csv['type'] == 'red', 'type'] = 1 
test_csv.loc[test_csv['type'] == 'white', 'type'] = 0

from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

y = LabelEncoder().fit_transform(y)
y = y.reshape(-1,1)
y = OneHotEncoder(sparse=False).fit_transform(y)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=333, train_size=0.8,
    stratify=y
)

print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
# (4397, 12) (1100, 12) (4397, 7) (1100, 7)

x = tf.compat.v1.placeholder(tf.float32, shape=[None,x_train.shape[1]],name='x')
y = tf.compat.v1.placeholder(tf.float32, shape=[None,y_train.shape[1]],name='y')

w = tf.compat.v1.Variable(tf.random_normal([x_train.shape[1],y_train.shape[1]]),name='weight')
b = tf.compat.v1.Variable(tf.zeros([1,y_train.shape[1]]),name='bias')

hypothesis = tf.nn.softmax(tf.add(tf.matmul(x,w),b))

loss_fn = tf.reduce_mean(-tf.reduce_mean(y*tf.log(hypothesis-y),axis=1)) # categorical_cross_entropy
loss_fn = tf.compat.v1.losses.softmax_cross_entropy(y,hypothesis)
# optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)

train = optimizer.minimize(loss_fn)

EPOCHS = 10000
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(1,EPOCHS+1):
        _, loss = sess.run([train,loss_fn],feed_dict={x:x_train,y:y_train})
        if step%100 == 0:
            print(f"{step}epo loss:{loss}")
        
    pred = sess.run(hypothesis,feed_dict={x:x_test,y:y_test})
    argmax_pred = np.argmax(pred,axis=1)
    # print(argmax_pred.shape)
argmax_y_test = np.argmax(y_test,axis=1)
# print(argmax_y_test.shape)    

from sklearn.metrics import accuracy_score
acc = accuracy_score(argmax_pred,argmax_y_test)
print("ACC: ",acc)

# ACC:  0.02727272727272727