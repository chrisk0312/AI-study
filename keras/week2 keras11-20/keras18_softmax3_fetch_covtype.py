import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
import time 
from keras.callbacks import EarlyStopping
from sklearn.metrics import  accuracy_score

#1.데이터
datasets = fetch_covtype()
x = datasets.data
y= datasets.target

print(x.shape,y.shape) #(581012, 54) (581012,)
print(y)
print(np.unique(y, return_counts=True))
print(pd.value_counts(y))
# 2    283301
# 1    211840
# 3     35754
# 7     20510
# 6     17367
# 5      9493
# 4      2747
#1 데이터
x = np.array(datasets.data)
y = np.array(datasets.target)

from sklearn.preprocessing import OneHotEncoder
y= y.reshape(-1,1) #(11)
ohe=OneHotEncoder(sparse=True) # 디폴트
y = ohe.fit_transform(y).toarray() #fit + transform
print(y)
print(y.shape)

x_train, x_test, y_train, y_test =  train_test_split(
    x,y, test_size=0.8, random_state=100,stratify=y)

print(x_train)
print(y_train)
print(x_test)
print(y_test)
print(np.unique(y_test,return_counts=True))



#2. 모델구성
model = Sequential()
model.add(Dense(10,input_dim=54)) 
model.add(Dense(20))
model.add(Dense(30))
model.add(Dense(40))
model.add(Dense(7, activation='softmax'))

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
model.fit(x_train, y_train, epochs=50, batch_size=48,
          validation_split=0.2,
          callbacks= [es],
          verbose=2)
end_time =time.time()

#4.평가,예측
result = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)
y_predict =np.argmax(y_predict, axis=1)
y_test = np.argmax(y_test, axis=1)
print("로스:",result[0])
print("acc :",result[1])
print("걸린시간:", round (end_time - start_time,2), "초")
acc =accuracy_score(y_predict,y_test)
print("acc :",acc)