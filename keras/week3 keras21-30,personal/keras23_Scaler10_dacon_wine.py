#https://dacon.io/competitions/open/235610/data

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from keras.callbacks import EarlyStopping
import time



#1. 데이터
path = "C:\\_data\\daicon\\wine\\"



train_csv = pd.read_csv(path + 'train.csv', index_col= 0)
print(train_csv)
test_csv = pd.read_csv(path + 'test.csv', index_col= 0)
print(test_csv)
submission_csv = pd.read_csv(path + "sample_submission.csv")
print(submission_csv)

print(train_csv.shape) #(5497, 13)
print(test_csv.shape) #(1000, 12)
print(submission_csv.shape) #(1000, 2)


print(train_csv.columns) #'quality', 'fixed acidity', 'volatile acidity', 'citric acid',
    #    'residual sugar', 'chlorides', 'free sulfur dioxide',
    #    'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol',
    #    'type'],
    
x = train_csv.drop(['quality'], axis= 1)
print(x)
y = train_csv['quality']

y = pd.get_dummies(y)


x.loc[x['type'] == 'red', 'type'] = 1
x.loc[x['type'] == 'white', 'type'] = 0
print(x)

test_csv.loc[test_csv['type'] == 'red', 'type'] = 1
test_csv.loc[test_csv['type'] == 'white', 'type'] = 0

print(test_csv)



#print(x.shape,y.shape) #(5497, 12) (5497, 7)




x_train, x_test, y_train, y_test = train_test_split(x, y, train_size= 0.8, shuffle= True, random_state=364, stratify= y)

from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
from sklearn.preprocessing import StandardScaler, RobustScaler

#mms = MinMaxScaler()
#mms = StandardScaler()
#mms = MaxAbsScaler()
mms = RobustScaler()

mms.fit(x_train)
x_train= mms.transform(x_train)
x_test= mms.transform(x_test)


#2. 모델구성

model = Sequential()
model.add(Dense(100, input_dim = 12))
model.add(Dense(90))
model.add(Dense(80))
model.add(Dense(70))
model.add(Dense(60))
model.add(Dense(20))
model.add(Dense(7, activation = 'softmax'))

x_train = np.asarray(x_train).astype(np.float32)
x_test = np.asarray(x_test).astype(np.float32)
test_csv = np.asarray(test_csv).astype(np.float32)


#3. 컴파일, 훈련
model.compile (loss = 'categorical_crossentropy', optimizer= 'adam', metrics= 'acc')
es = EarlyStopping(monitor= 'val_acc', mode = 'max', patience= 200, verbose= 0 ,restore_best_weights= False)

start_time = time.time()
model.fit(x_train, y_train, epochs = 2000, batch_size= 10, validation_split= 0.4, verbose=2, callbacks= [es])
end_time = time.time()

#4. 평가, 예측

results = model.evaluate(x_test, y_test)
print("로스 :", results[0])
print("acc :", results[1])
y_predict = model.predict(x_test)

print(y_predict)
print(y_predict.shape, y_test.shape) #(1650, 2) (1650, 2)

y_submit = model.predict(test_csv)


y_test = np.argmax(y_test, axis=1) #원핫을 통과해서 아그맥스를 다시 통과시켜야함
y_predict = np.argmax(y_predict, axis=1 )
y_submit = np.argmax(y_submit, axis=1 )+3
print(y_test, y_predict)
result = accuracy_score(y_test, y_predict)
submission_csv['quality'] = y_submit
acc = accuracy_score(y_predict, y_test)
ltm = time.localtime(time.time())
print("acc :", acc)
print("로스 :", results[0])
save_time = f"{ltm.tm_year}{ltm.tm_mon}{ltm.tm_mday}{ltm.tm_hour}{ltm.tm_min}{ltm.tm_sec}" 
submission_csv.to_csv(path+f"submission_{save_time}e.csv", index=False)




print("걸린 시간 :", round(end_time - start_time, 2), "초" )
print(np.unique(y_submit))

#minmax
#acc : 0.5581818181818182
#로스 : 1.0659217834472656

#StandardScaler
#acc : 0.5681818181818182
#로스 : 1.0569180250167847

#MaxAbsScaler()
#acc : 0.5709090909090909
#로스 : 1.0683833360671997

#mms = RobustScaler()
#acc : 0.5609090909090909
#로스 : 1.0590828657150269