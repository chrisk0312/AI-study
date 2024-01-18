#https://dacon.io/competitions/open/235610/overview/description
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
from sklearn.preprocessing import OneHotEncoder
import time
from sklearn.preprocessing import LabelEncoder

#1 데이터

path = "c:\_data\dacon\wine\\"
oe=OneHotEncoder(sparse=False)
train_csv = pd.read_csv(path + "train.csv", index_col =0)
print(train_csv)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
print(test_csv)
submission_csv = pd.read_csv(path + "sample_submission.csv")
print(submission_csv)

print(train_csv.shape) #(5497, 13)
print(test_csv.shape) #(1000, 12)
print(submission_csv.shape) #(1000, 2)
print(train_csv.columns) 
# Index(['quality', 'fixed acidity', 'volatile acidity', 'citric acid',
#        'residual sugar', 'chlorides', 'free sulfur dioxide',
#        'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol',
#        'type'],
#       dtype='object')

print(train_csv.info())
print(test_csv.info())
print(train_csv.describe())


train_csv['type'] = train_csv['type'].map({'white':1,'red':0}).astype(int)
test_csv['type'] = test_csv['type'].map({'white':1,'red':0}).astype(int)

x= train_csv.drop (['quality'], axis=1)
print(x)
y = train_csv['quality']
print(y)

print(np.unique(y, return_counts=True)) 
#(array([3, 4, 5, 6, 7, 8, 9], dtype=int64), array([  26,  186, 1788, 2416,  924,  152,    5], dtype=int64))

y= pd.get_dummies(y)

import pandas as pd


print(train_csv)
print(test_csv)


x_train,x_test, y_train, y_test = train_test_split( x,y, train_size=0.7,shuffle=True, random_state=100, stratify= y)
print(x_train.shape,x_test.shape) #(3847, 12) (1650, 12)
print(y_train.shape, y_test.shape) #(3847, 7) (1650, 7)


#2. 모델구성
model = Sequential()
model.add(Dense(64,input_dim=12, activation= 'relu')) 
model.add(Dense(128, activation= 'relu'))
model.add(Dense(64,activation= 'relu'))
model.add(Dense(7, activation='softmax'))

x_train = np.asarray(x_train).astype(np.float32)
x_test = np.asarray(x_test).astype(np.float32)
test_csv = np.asarray(test_csv).astype(np.float32)

#3 컴파일,훈련
model.compile(loss = 'categorical_crossentropy', optimizer ='adam',
              metrics =['acc'])
start_time =time.time()
es = EarlyStopping(monitor ='val_loss',
                   mode = 'min',
                   patience = 100,
                   verbose=2,
                   restore_best_weights= True,
                  )

model.fit(x_train, y_train, epochs=15000, batch_size=15,
          validation_split=0.4,
          callbacks= [es],
          verbose=2)
end_time =time.time()



#4.평가,예측

results = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)
y_submit = model.predict(test_csv)

y_test = np.argmax(y_test, axis=1)
y_predict =np.argmax(y_predict, axis=1)
y_submit =np.argmax(y_submit, axis=1)+3


result =accuracy_score(y_test,y_predict)
submission_csv['quality'] = y_submit

def ACC(a,b):
   return accuracy_score(a,b)
acc = ACC(y_test,y_predict)
print('score',acc)

print(submission_csv)
print(submission_csv.shape)
submission_csv.to_csv(path+"submission_0112_4.csv", index=False)


print("acc :",result)
print("로스:",results[0])
print("acc :",results[1])
print("걸린시간:", round (end_time - start_time,2), "초")
