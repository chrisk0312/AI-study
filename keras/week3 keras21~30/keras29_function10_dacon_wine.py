from keras.models import Sequential, Model
from keras.layers import Dense, Input
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
import sklearn.preprocessing

path = "C:\\_data\\DACON\\와인품질분류\\"
train_csv = pd.read_csv(path+"train.csv",index_col=0)
test_csv = pd.read_csv(path+"test.csv",index_col=0)
submit_csv = pd.read_csv(path+"sample_submission.csv")

# print(train_csv.isna().sum(),test_csv.isna().sum()) 결측치 존재안함

# print(train_csv,test_csv,sep='\n') #[5497 rows x 13 columns], [1000 rows x 12 columns]
x = train_csv.drop(['quality'],axis=1)
y = train_csv['quality']

print(np.unique(y,return_counts=True))

# print(x.shape,y.shape)  #(5497, 12) (5497,)
print(np.unique(y,return_counts=True))
y = pd.get_dummies(y)
# print(y.shape)          #(5497, 7)

x.loc[x['type'] == 'red', 'type'] = 1
x.loc[x['type'] == 'white', 'type'] = 0
# print(x)
test_csv.loc[test_csv['type'] == 'red', 'type'] = 1 
test_csv.loc[test_csv['type'] == 'white', 'type'] = 0
# print(test_csv)
acc = 0
r = int(np.random.uniform(1,1000))
# r = 894
x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.8,random_state=r,stratify=y)

from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler
# scaler = MinMaxScaler().fit(x_train)    #최솟값을 0 최댓값을 1로 스케일링
# scaler = StandardScaler().fit(x_train)  #정규분포로 바꿔줘서 스케일링
# scaler = MaxAbsScaler().fit(x_train)    #
scaler = RobustScaler().fit(x_train)    #

x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
test_csv = scaler.transform(test_csv)

#model
# model = Sequential()
# model.add(Dense(512,input_dim=12, activation='relu'))
# model.add(Dense(256, activation='relu'))
# model.add(Dense(128, activation='relu'))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(16, activation='relu'))
# model.add(Dense(7, activation='softmax'))
input1 = Input(shape=(12,))
d1 = Dense(512, activation='relu')(input1)
d2 = Dense(256, activation='relu')(d1)
d3 = Dense(128, activation='relu')(d2)
d4 = Dense(64, activation='relu')(d3)
d5 = Dense(32, activation='relu')(d4)
d6 = Dense(16, activation='relu')(d5)
output1 = Dense(7,activation='softmax')(d6)
model = Model(inputs=input1,outputs=output1)

x_train =np.asarray(x_train).astype(np.float32) #Numpy는 기본적으로 float32 연산이기 때문에 되도록 맞춰주는게 좋다
x_test =np.asarray(x_test).astype(np.float32)
test_csv =np.asarray(test_csv).astype(np.float32)

# for row in x_train:
#     print(row)

#compile & fit
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['acc'])
es = EarlyStopping(monitor='val_loss',mode='auto',patience=200,restore_best_weights=True,verbose=1)
from keras.callbacks import ModelCheckpoint
mcp = ModelCheckpoint(monitor='val_loss',mode='min',save_best_only=True,
                    filepath="c:/_data/_save/MCP/dacon_wine"+"{epoch:04d}{val_loss:.4f}.hdf5")
print(x_train.shape,y_train.shape)
hist = model.fit(x_train,y_train,epochs=4096,batch_size=64,validation_split=0.3,verbose=2,callbacks=[es,mcp])

#evaluate & predict
loss = model.evaluate(x_test,y_test,verbose=0)
y_predict = np.argmax(model.predict(x_test),axis=1)
y_submit = np.argmax(model.predict(test_csv),axis=1)+3
y_test = np.argmax(y_test,axis=1)

print(f"{r=} \nLOSS: {loss[0]}\n ACC:  {accuracy_score(y_test,y_predict)}({loss[1]} by loss[1])")
import time
time.sleep(1.5)

acc = loss[1]

print(np.unique(y_submit,return_counts=True))

print(y_test.shape,y_predict.shape)

#y submit
submit_csv['quality'] = y_submit
import datetime
dt = datetime.datetime.now()
submit_csv.to_csv(path+f"wine_{dt.day}day{dt.hour:2}{dt.minute:2}_ACC{loss[1]:.4f}.csv",index=False)

#그래프 출력
plt.figure(figsize=(12,9))
plt.title("DACON Wine Classification")
plt.xlabel('epochs')
plt.ylabel('loss')
plt.plot(hist.history['loss'],label='loss',color='red')
plt.plot(hist.history['val_loss'],label='val_loss',color='blue')
plt.legend()
# plt.show()

# r=894
# LOSS: 2.3342490196228027
#  ACC:  0.5781818181818181(0.578181803226471 by loss[1])

# r=912
# LOSS: 1.0722206830978394
#  ACC:  0.5536363636363636(0.553636372089386 by loss[1])

# r=20
# LOSS: 1.0789892673492432
#  ACC:  0.5509090909090909(0.5509091019630432 by loss[1])

# r=433
# LOSS: 1.0544521808624268
#  ACC:  0.5572727272727273(0.557272732257843 by loss[1])

# MinMaxScaler
# LOSS: 1.0109286308288574
#  ACC:  0.5690909090909091(0.5690909028053284 by loss[1])

# StandardScaler
# LOSS: 1.0394665002822876
#  ACC:  0.5745454545454546(0.5745454430580139 by loss[1])

# MaxAbsScaler
# LOSS: 1.0278842449188232
#  ACC:  0.56(0.5600000023841858 by loss[1])

# RobustScaler
# LOSS: 1.058943510055542
#  ACC:  0.5636363636363636(0.5636363625526428 by loss[1])