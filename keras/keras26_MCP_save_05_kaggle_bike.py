import numpy as np
import pandas as pd #판다스에 데이터는 넘파이 형태로 들어가있음.
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_squared_log_error, accuracy_score
import time
from keras.callbacks import EarlyStopping

#1. 데이터
path = "C:\\_data\\kaggle\\bike\\"


train_csv = pd.read_csv(path + "train.csv", index_col=0)
print(train_csv)
test_csv = pd.read_csv(path + "test.csv", index_col= 0)
print(test_csv)
submission_csv = pd.read_csv(path + "sampleSubmission.csv")
print(submission_csv)

print(train_csv.shape)      # (10886, 11)
print(test_csv.shape)       # (6493, 8)
print(submission_csv.shape) # (6493, 2)

print(train_csv.columns)
# ['season', 'holiday', 'workingday', 'weather', 'temp', 'atemp',
#       'humidity', 'windspeed', 'casual', 'registered', 'count']

print(train_csv.info())
print(test_csv.info())


x = train_csv.drop(['casual','registered','count'], axis=1)
print(x)
y = train_csv['count']
print(y)

print(train_csv.index)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size= 0.77, shuffle = False, random_state=1266)
print(x_train.shape, x_test.shape) #(7620, 8) (3266, 8)
print(y_train.shape, y_test.shape) #(7620, ) (3266, )


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
model.add(Dense(1024, input_dim = 8))
model.add(Dense(512,))
model.add(Dense(256,))
model.add(Dense(8, activation='relu'))
model.add(Dense(1,  activation='relu'))

#3. 컴파일, 훈련


end_time = time.time()

from keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor = 'val_loss', mode = 'min', patience = 400, verbose = 0, restore_best_weights= True)
mcp = ModelCheckpoint(monitor='val_loss', mode = 'auto', verbose= 1, save_best_only=True, filepath='../_data/_save/MCP/keras26_05_MCP.hdf5')

model.compile(loss= 'mse', optimizer= 'adam', metrics='mse' ) #mae 2.64084 r2 0.8278   mse 12.8935 r2 0.82
hist = model.fit(x_train, y_train, callbacks=[es,mcp], epochs= 1200, batch_size = 100, validation_split= 0.27)





#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
y_submit = model.predict(test_csv)


#print(y_submit)
#print(y_submit.shape) #(6493, 1)
#print(submission_csv.shape) #(6493, 2)




#print("걸린 시간:", round(end_time - start_time, 2), "초")

submission_csv['count'] = y_submit

print(submission_csv)
accuracy_score = ((y_test, y_submit))

y_submit = (y_submit.round(0).astype(int))


#submission_csv.to_csv(path + "submission_29.csv", index= False)
print("음수갯수 :", submission_csv[submission_csv['count']<0].count())
print("로스 :", loss)
print("R2 스코어 :", r2)
print("정확도 :",accuracy_score)

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
rmse = RMSE(y_test, y_predict)
print("RMSE : ", rmse)


def RMSLE(y_test, y_predict):
    return np.sqrt(mean_squared_log_error(y_test, y_predict))
rmsle = RMSLE(y_test, y_predict)
print("로스 :", loss)
print("RMSLE :", rmsle)

import time as tm
ltm = tm.localtime(tm.time())
save_time = f"{ltm.tm_year}{ltm.tm_mon}{ltm.tm_mday}{ltm.tm_hour}{ltm.tm_min}{ltm.tm_sec}" 
submission_csv.to_csv(path + f"submission_{save_time}{rmsle:.3f}.csv", index=False)
#MSE : 23175.111328125
#R2 스코어 : 0.27044473122031987
#RMSE :  152.23374956711748
#RMSLE : 1.3152084898668681

import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus']= False
plt.figure(figsize= (9,6))
plt.plot(hist.history['loss'], c = 'red', label = 'loss', marker = '.')
plt.plot(hist.history['val_loss'], c = 'blue', label = 'val_loss', marker = '.')
plt.legend(loc = 'upper right')
plt.title("케글바이크 LOSS")
plt.xlabel('epoch')
plt.grid()
plt.show()


#True
#MSE : 42507.94921875
#R2 스코어 : 0.107033535623137
#RMSE :  206.1745593621529
#RMSLE : 1.2849059435800314

#False
#MSE : 44973.4375
#R2 스코어 : 0.055240887807962324
#RMSE :  212.06941832566426
#RMSLE : 1.2467025777594747

#
#MSE : 22668.6015625
#R2 스코어 : 0.36016960386561647
#RMSE :  150.5609694781289
#RMSLE : 1.3219243059301249

#mms = MinMaxScaler()
#로스 : [40612.984375, 40612.984375]
#RMSLE : 1.2766554161780495

#mms = StandardScaler()
#로스 : [43217.55859375, 43217.55859375]
#RMSLE : 1.2211611916371106

#mms = MaxAbsScaler()
#로스 : [43217.55859375, 43217.55859375]
#RMSLE : 1.2211611916371106

#mms = RobustScaler()
#로스 : [45134.109375, 45134.109375]
#RMSLE : 1.2113118872869002

#RMSE :  199.78955029836519
#로스 : [39915.8671875, 39915.8671875]
#RMSLE : 1.225977399907451