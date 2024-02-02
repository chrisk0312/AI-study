#https://dacon.io/competitions/open/235610/data

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
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
model.add(LSTM(100, input_shape = (12,1)))
model.add(Dropout(0.3))
model.add(Dense(90))
model.add(Dense(80))
model.add(Dropout(0.2))
model.add(Dense(70))
model.add(Dense(60))
model.add(Dropout(0.1))
model.add(Dense(20))
model.add(Dense(7, activation = 'softmax'))

x_train = np.asarray(x_train).astype(np.float32)
x_test = np.asarray(x_test).astype(np.float32)
test_csv = np.asarray(test_csv).astype(np.float32)



# #3. 컴파일, 훈련
import datetime
date= datetime.datetime.now()
# print(date) #2024-01-17 11:00:58.591406
# print(type(date)) #<class 'datetime.datetime'>
date = date.strftime("%m%d-%H%M") #m=month, M=minutes
# print(date) #0117_1100
# print(type(date)) #<class 'str'>

path= 'c:/_data/_save/MCP/_k28/' #경로(스트링data (문자))
filename = '{epoch:04d}-{val_loss:.4f}.hdf5' #filename= 에포4자리수-발로스는 소숫점4자리까지 표시. 예)1000-0.3333.hdf5
filepath = "".join([path, 'k28_8-', date, "_", filename]) #""공간에 ([])를 합쳐라.



from keras.callbacks import EarlyStopping, ModelCheckpoint
# es = EarlyStopping(monitor = 'val_acc', mode = 'max', patience = 200, verbose = 0, restore_best_weights= True)
mcp = ModelCheckpoint(monitor='val_acc', mode = 'max', verbose= 1, save_best_only=True, filepath=filepath)
model.compile(loss= 'categorical_crossentropy', optimizer= 'adam', metrics= 'acc' ) #mae 2.64084 r2 0.8278   mse 12.8935 r2 0.82
start_time = time.time()
hist = model.fit(x_train, y_train, callbacks=[mcp], epochs= 2000, batch_size = 10, validation_split= 0.4)
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
save_time = f"{ltm.tm_year}{ltm.tm_mon}{ltm.tm_mday}{ltm.tm_hour}{ltm.tm_min}{ltm.tm_sec}" 
submission_csv.to_csv(path+f"submission_{save_time}e.csv", index=False)




print("걸린 시간 :", round(end_time - start_time, 2), "초" )
print(np.unique(y_submit))



#mms = MinMaxScaler()
#로스 : 1.0762532949447632
#acc : 0.5527272727272727

#mms = StandardScaler()
#로스 : 1.060926079750061
#acc : 0.5681818181818182

#mms = MaxAbsScaler()
#로스 : 1.0695488452911377
#acc : 0.5581818181818182

#mms = RobustScaler()
#로스 : 1.0639983415603638
#acc : 0.5636363636363636

#로스 : 1.0716124773025513
#acc : 0.5681818181818182

#로스 : 1.0631638765335083
#acc : 0.5654545454545454



#로스 : 1.0702488422393799
#acc : 0.5654545454545454