from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import accuracy_score, r2_score
import pandas as pd
import numpy as np

path = "C:\\_data\\DACON\\와인품질분류\\"
train_csv = pd.read_csv(path+"train.csv",index_col=0)
test_csv = pd.read_csv(path+"test.csv",index_col=0)
submit_csv = pd.read_csv(path+"sample_submission.csv")

# print(train_csv.isna().sum(),test_csv.isna().sum()) 결측치 존재안함

# print(train_csv,test_csv,sep='\n') #[5497 rows x 13 columns], [1000 rows x 12 columns]
x = train_csv.drop(['quality'],axis=1)
y = train_csv['quality']

print(np.unique(y,return_counts=True))
# (array([3, 4, 5, 6, 7, 8, 9], dtype=int64), array([  26,  186, 1788, 2416,  924,  152,    5], dtype=int64))

x.loc[x['type'] == 'red', 'type'] = 1
x.loc[x['type'] == 'white', 'type'] = 0
# print(x)
test_csv.loc[test_csv['type'] == 'red', 'type'] = 1 
test_csv.loc[test_csv['type'] == 'white', 'type'] = 0

from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

y = LabelEncoder().fit_transform(y)

import tensorflow as tf
import numpy as np
r = np.random.randint(1,1000)
# r = 8603
tf.compat.v1.set_random_seed(r)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder
y = OneHotEncoder(sparse=False).fit_transform(y.reshape(-1,1))
x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.8,random_state=r)
scaler = StandardScaler().fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

print(x_train.shape,y_train.shape,x_test.shape,y_test.shape)
# (4397, 12) (4397, 7) (1100, 12) (1100, 7)

x = tf.compat.v1.placeholder(tf.float32,shape=[None,x_train.shape[1]])
y = tf.compat.v1.placeholder(tf.float32,shape=[None,y_train.shape[1]])

class Dense_layer():
    def __init__(self, output_dim, input_dim, activation=None) -> None:
        self.w = tf.compat.v1.Variable(tf.random_normal([input_dim,output_dim]))
        self.b = tf.compat.v1.Variable(tf.zeros([1,output_dim]),dtype=tf.float32)
        self.activation = activation
    def get_layer(self,x):
        result = tf.matmul(x,self.w) + self.b
        if self.activation is not None:
            return self.activation(result)
        return result


layer1 = Dense_layer(512,x_train.shape[1],tf.nn.relu).get_layer(x)
layer2 = Dense_layer(256,512,tf.nn.relu).get_layer(layer1)
layer3 = Dense_layer(256,256,tf.nn.sigmoid).get_layer(layer2)
layer4 = Dense_layer(128,256,tf.nn.relu).get_layer(layer3)
layer5 = Dense_layer(64,128,tf.nn.relu).get_layer(layer4)
layer6 = Dense_layer(32,64,tf.nn.sigmoid).get_layer(layer5)
hypothesis = Dense_layer(y_train.shape[1],32,tf.nn.softmax).get_layer(layer6)

loss_fn = tf.compat.v1.losses.softmax_cross_entropy(y,hypothesis)
optimizer = tf.train.AdamOptimizer(learning_rate=0.001)

train = optimizer.minimize(loss_fn)

EPOCHS = 3000
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(1,EPOCHS+1):
        _, loss = sess.run([train,loss_fn],feed_dict={x:x_train,y:y_train})
        if step%100 == 0:
            print(f"{step}epo loss={loss}")
        
    pred = sess.run(hypothesis,feed_dict={x:x_test})
    pred = np.around(pred)
    
print("pred: ",pred)
from sklearn.metrics import accuracy_score
acc = accuracy_score(pred,y_test)

print("ACC: ",acc)
print("Random state: ",r)

# ACC:  0.5527272727272727
# Random state:  410