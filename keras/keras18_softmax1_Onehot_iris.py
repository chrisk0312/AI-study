import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import time 
from keras.callbacks import EarlyStopping
from sklearn.metrics import  accuracy_score
import tensorflow as tf

#1.데이터
datasets = load_iris()
print(datasets)
print(datasets.DESCR)
print(datasets.feature_names)
x = datasets.data
y = datasets.target
print(x.shape, y.shape) #(150, 4) (150,)
print(y)
print(np.unique(y, return_counts=True))
#(array([0, 1, 2]), array([50, 50, 50], dtype=int64))
print(pd.value_counts(y))
# 0    50
# 1    50
# 2    50

# print("=====================onehot1================")


# #1 onehot 1. 케라스
# from keras.utils import to_categorical
# y_ohe = to_categorical(y)
# print(y_ohe)
# print(y_ohe.shape)# (150,3)

# print("=====================onehot2================")

# #2 onehot 2 , 판다스
# y_ohe2 = pd.get_dummies(y)
# print(y_ohe2)
# print(y_ohe.shape)

print("=====================onehot3================")

# #3 onehot3, 사이킷런
# from sklearn.preprocessing import OneHotEncoder
# y=y.reshape(-1,1) #(150, 1)
# y= y.reshape(150,1) #(150, 1)

# ohe=OneHotEncoder(sparse=True) # 디폴트
# # ohe.fit(y)
# # y_ohe3 = ohe.transform(y)
# # y = ohe.fit_transform(y) #fit + transform
# y = ohe.fit_transform(y).toarray() #fit + transform
# print(y)
# print(y.shape)


#1 데이터
x = np.array(datasets.data)
y = np.array(datasets.target)
#y = pd.get_dummies(y)
from sklearn.preprocessing import OneHotEncoder
y= y.reshape(-1,1) #(150, 1)
#y= y.reshape(150,1) #(150, 1)

ohe=OneHotEncoder(sparse=True) # 디폴트
# ohe.fit(y)
# y_ohe3 = ohe.transform(y)
# y = ohe.fit_transform(y) #fit + transform
y = ohe.fit_transform(y).toarray() #fit + transform
print(y)
print(y.shape)



x_train, x_test, y_train, y_test =  train_test_split(
    x,y, test_size=0.7, random_state=100,stratify=y)

print(x_train)
print(y_train)
print(x_test)
print(y_test)


#2. 모델구성
model = Sequential()
model.add(Dense(10,input_dim=4)) 
model.add(Dense(20))
model.add(Dense(30))
model.add(Dense(40))
model.add(Dense(3, activation='softmax'))

#3 컴파일,훈련
model.compile(loss = 'categorical_crossentropy', optimizer ='adam',
              metrics =['acc'])
start_time =time.time()
es = EarlyStopping(monitor ='val_loss',
                   mode = 'min',
                   patience = 30,
                   verbose=2,
                   restore_best_weights= True,
                )
model.fit(x_train, y_train, epochs=300, batch_size=1,
          validation_split=0.3,
          callbacks= [es],
          verbose=2)
end_time =time.time()

#4.평가,예측
loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)
y_predict = np.round(y_predict)


print("걸린시간:", round (end_time - start_time,2), "초")
