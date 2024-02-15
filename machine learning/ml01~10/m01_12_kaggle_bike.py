from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_squared_log_error
from sklearn.model_selection import train_test_split
import pandas as pd
import datetime
import time
import math
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping
from sklearn.svm import LinearSVR

#data
path = "C:\\_data\\KAGGLE\\bike-sharing-demand\\"
train_csv = pd.read_csv(path+"train.csv",index_col=0)
test_csv = pd.read_csv(path+"test.csv",index_col=0)
submission_csv = pd.read_csv(path+"sampleSubmission.csv")

x = train_csv.drop(['casual','registered','count'],axis=1)
y = train_csv['count']

print(x.shape, y.shape)

r = int(np.random.uniform(1,1000))
r=2
x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.7,shuffle=False,random_state=r)

from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler
# scaler = MinMaxScaler().fit(x_train)    #최솟값을 0 최댓값을 1로 스케일링
# scaler = StandardScaler().fit(x_train)  #정규분포로 바꿔줘서 스케일링
# scaler = MaxAbsScaler().fit(x_train)    #
scaler = RobustScaler().fit(x_train)    #

x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
test_csv = scaler.transform(test_csv)

# x_train, x_val, y_train, y_val = train_test_split(x_train,y_train,train_size=0.7,shuffle=False)

print(f"{x_train.shape=},{x_test.shape=}")


#model
model = LinearSVR(C=100)

#compile & fit
model.fit(x_train,y_train)

#evaluate & predict 
loss = model.score(x_test,y_test )
y_predict = model.predict(x_test)
r2 = r2_score(y_test,y_predict)
y_submit = model.predict(test_csv)

print(f"{r=}\n{loss=}\n{r2=}")
time.sleep(1.5)

y_submit = np.where(y_submit < 0, 0, y_submit)
y_predict = np.where(y_predict < 0, 0, y_predict)
print(np.where(y_submit<0))
print(np.where(y_test<0))

#### CSV파일 생성 ####
submission_csv['count'] = y_submit
dt = datetime.datetime.now()
# submission_csv.to_csv(path+f"submission_{dt.day}day{dt.hour}-{dt.minute}.csv",index=False)
submission_csv.to_csv(path+f"submission_{dt.day}day{dt.hour}-{dt.minute}_loss{loss}.csv",index=False)


#### 음수 개수와 RMSLE출력 ####
num_of_minus = submission_csv[submission_csv['count']<0].count()
print("num of minus",num_of_minus['count'])

def RMSLE(y_test,y_predict):
    return np.sqrt(mean_squared_log_error(y_test,y_predict))

if num_of_minus['count'] == 0:    
    print("RMSLE: ",RMSLE(y_test,y_predict))
else:
    print("음수갯수: ",num_of_minus['count'])
    for i in range(len(y_submit)):
        if y_submit[i] < 0:
            y_submit[i] = 0
    
# plt.rcParams['font.family'] = 'Malgun Gothic'
# plt.title('kaggle bike')
# plt.xlabel('epochs')
# plt.ylabel('loss')
# plt.plot(hist.history['loss'],label='loss',color='red',marker='.')
# plt.plot(hist.history['val_loss'],label='val_loss',color='blue',marker='.')
# plt.grid()
# plt.legend()
# plt.show()

# r=662
# loss=23495.251953125
# r2=0.2603666422772616
# RMSLE:  1.2877634377136509

# r=662
# loss=22796.43359375
# r2=0.28236574526494185
# RMSLE:  1.3233963468270995

# loss=40385.41015625
# r2=0.15680798684173702
# RMSLE:  1.286986737279555

# loss=43231.234375
# r2=0.09739137590805513
# RMSLE:  1.2574265328392626

# MinMaxScaler
# loss=[38606.78125, 38606.78125]
# r2=0.19394372959029815
# RMSLE:  1.2225483246037947

# StandardScaler
# loss=[40058.3203125, 40058.3203125]
# r2=0.16363740670246008
# RMSLE:  1.2787136000962247

# MaxAbsScaler
# loss=[39243.7109375, 39243.7109375]
# r2=0.1806453152200005
# RMSLE:  1.2434084374789507

# RobustScaler
# loss=[39731.94921875, 39731.94921875]
# r2=0.17045145049558763
# RMSLE:  1.2225427797423805

# LinearSVR
# RMSLE:  1.256727136597206