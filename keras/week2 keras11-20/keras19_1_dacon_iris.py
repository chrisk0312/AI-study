# https://dacon.io/competitions/open/236070/data

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

#1 데이터

path = "c:\_data\dacon\iris\\"
train_csv = pd.read_csv(path + "train.csv", index_col =0)
print(train_csv)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
print(test_csv)
submission_csv = pd.read_csv(path + "sample_submission.csv")
print(submission_csv)

print(train_csv.shape) #(120, 5)
print(test_csv.shape) #(30, 4)
print(submission_csv.shape) #(30, 2)
print(train_csv.columns) 
# #Index(['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)',
#        'petal width (cm)', 'species'],
#       dtype='object')
print(train_csv.info())
print(test_csv.info())
print(train_csv.describe())

x= train_csv.drop(['species'], axis=1) 
print(x)
y = train_csv['species']
print(y)

print(np.unique(y, return_counts=True))

y= pd.get_dummies(y)

x_train,x_test, y_train, y_test = train_test_split( x,y, train_size=0.7,shuffle=True, random_state=100, stratify= y)
print(x_train.shape,x_test.shape) #(24, 4) (96, 4)
print(y_train.shape, y_test.shape) #(24,) (96,)

#2. 모델구성
model = Sequential()
model.add(Dense(8,input_dim=4)) 
model.add(Dense(16))
model.add(Dense(32))
model.add(Dense(64))
model.add(Dense(3, activation='softmax'))

#3 컴파일,훈련
model.compile(loss = 'categorical_crossentropy', optimizer ='adam',
              metrics =['acc'])
start_time =time.time()
es = EarlyStopping(monitor ='val_loss',
                   mode = 'min',
                   patience = 300,
                   verbose=2,
                   restore_best_weights= True,
                )
model.fit(x_train, y_train, epochs=500, batch_size=1,
          validation_split=0.2,
          callbacks= [es],
          verbose=2)
end_time =time.time()



#4.평가,예측

results = model.evaluate(x_test, y_test)
y_submit = model.predict(test_csv)
y_predict = model.predict(x_test)
y_predict =np.argmax(y_predict, axis=1)
y_submit =np.argmax(y_submit, axis=1)

y_test = np.argmax(y_test, axis=1)
result =accuracy_score(y_test,y_predict)
submission_csv['species'] = y_submit

print(submission_csv)
print(submission_csv.shape)
submission_csv.to_csv(path+"submission_0112_4.csv", index=False)

print("acc :",result)
print("로스:",results[0])
print("acc :",results[1])
print("걸린시간:", round (end_time - start_time,2), "초")
