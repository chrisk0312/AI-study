#https://www.kaggle.com/competitions/bike-sharing-demand/data?select=train.csv

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error,mean_squared_log_error
import time
from keras.callbacks import EarlyStopping

#1 데이터

path = "c:\_data\kaggle\\bike\\"
train_csv = pd.read_csv(path + "train.csv",index_col=0)
print(train_csv)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
print(test_csv)
submission_csv = pd.read_csv(path +"sampleSubmission.csv",)
print(submission_csv)
print(train_csv.shape) #(10886, 11)
print(test_csv.shape) #(6493, 8)
print(submission_csv.shape) #(6493, 2)
print(train_csv.columns) #Index(['season', 'holiday', 'workingday', 'weather', 'temp', 'atemp',
                         #'humidity', 'windspeed', 'casual', 'registered', 'count'],
print(train_csv.info())
print(test_csv.info())
print(train_csv.describe())

###########x와 y의 값을 분리
x= train_csv.drop(['casual', 'registered', 'count'], axis=1) 
print(x)
y = train_csv['count']
print(y)

x_train,x_test, y_train, y_test = train_test_split(
    x,y, train_size=0.9,shuffle=False, random_state=100,)
print(x_train.shape,x_test.shape) #(2177, 8) (8709, 8)
print(y_train.shape, y_test.shape) #(2177,) (8709,)

#2. 모델
model = Sequential()
model.add(Dense(32,input_dim = 8,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(128,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(1))

#3.컴파일, 훈련
model.compile(loss='mse', optimizer= 'adam')
start_time = time.time()
es = EarlyStopping(monitor = 'val_loss',
                   mode = min,
                   patience=100,
                   verbose=2,
                   restore_best_weights = True,
                   )
hist = model.fit(x_train, y_train, epochs=300, batch_size=32,
          validation_split=0.3,
          callbacks= [es],
          verbose=2)
end_time = time.time()

#4. 평가,예측
loss = model.evaluate(x_test, y_test)
y_submit = model.predict(test_csv)
y_predict = model.predict(x_test)
print(y_submit)
print(y_submit.shape) # (6493, 1)


print("걸린시간:", round)
print("===========hist==========")
print(hist)
print("===========hist==========")
print(hist.history)
print("===========hist==========")
print(hist.history['loss'])
print("===========hist==========")
print(hist.history['val_loss'])
print("===========hist==========")


print('===================================')
submission_csv['count'] = y_submit
print(submission_csv)
print(submission_csv.shape) #(6493, 2)
submission_csv.to_csv(path+"submission_0109_06.csv", index= False)
print("MSE :", loss)

################################
print("음수갯수:", submission_csv[submission_csv['count']<0].count())
y_predict = model.predict(x_test)
r2 =r2_score(y_test,y_predict)
def RMSE(y_test, y_predict):
    return(np.sqrt(mean_squared_error(y_test,y_predict)))
rmse =RMSE(y_test, y_predict)
print("RMSE :",rmse)
print("R2 스코어:", r2)
print("걸린시간:", round (end_time - start_time,2), "초")

import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.figure(figsize=(10,10))
plt.plot(hist.history['loss'],c='red',label = 'loss',marker=".")
plt.plot(hist.history['val_loss'], c='blue',label='val_loss',marker ='.' )
plt.legend(loc='upper right')
plt.title('따릉이 loss')
plt.xlabel('에포')
plt.ylabel('로스')
plt.grid()
plt.show()

# RMSE : 202.09263788047966
# R2 스코어: -0.083854081301584

# RMSE : 185.66880853811065
# R2 스코어: 0.08515454907600273
