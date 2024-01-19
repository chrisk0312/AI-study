from keras.models import Sequential,Model
from keras.layers import Dense,Dropout,Input
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.datasets import fetch_covtype
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score

datasets = fetch_covtype()
x = datasets.data  
y = datasets.target

# print(x.shape,y.shape)      #(581012, 54) (581012,)
# print(pd.value_counts(y))
# 2    283301
# 1    211840
# 3     35754
# 7     20510
# 6     17367
# 5      9493
# 4      2747

# y = pd.get_dummies(y)
# ohe = OneHotEncoder(sparse=False)
# y = ohe.fit_transform(y.reshape(-1,1))
print(np.unique(y,return_counts=True))
y = to_categorical(y)

# print(y,y.shape,sep='\n')
# print(np.count_nonzero(y[:,0]))
'''
sklearn : (581012, 7)
pandas  : (581012, 7)
keras   : (581012, 8)
keras 첫번째 열이 미심직어 찍어보니
print(np.count_nonzero(y[:,0])) # 0
따라서 첫번째 열 잘라내고 슬라이싱
'''
# print(y.shape)

y = y[:,1:]
# print(y,y.shape,sep='\n')       # (581012, 7)
# print(np.count_nonzero(y[:,0])) # 211840

r = int(np.random.uniform(1,1000))
x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.7,random_state=r,stratify=y)

from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler
# scaler = MinMaxScaler().fit(x_train)    #최솟값을 0 최댓값을 1로 스케일링
# scaler = StandardScaler().fit(x_train)  #정규분포로 바꿔줘서 스케일링
# scaler = MaxAbsScaler().fit(x_train)    #
scaler = RobustScaler().fit(x_train)    #

x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#model
# model = Sequential()
# model.add(Dense(512,input_dim=54,activation='relu'))
# model.add(Dense(256, activation='relu'))
# model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.7))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(16, activation='relu'))
# model.add(Dense(8,  activation='relu'))
# model.add(Dense(7,  activation='softmax'))

input = Input(shape=(54,))
d1 = Dense(512, activation='relu')(input)
d2 = Dense(256, activation='relu')(d1)
d3 = Dense(128, activation='relu')(d2)
dr1 = Dropout(0.7)(d3)
d4 = Dense(64, activation='relu')(dr1)
d5 = Dense(32, activation='relu')(d4)
d6 = Dense(16, activation='relu')(d5)
d7 = Dense(8, activation='relu')(d6)
output = Dense(7, activation='softmax')(d7)

model = Model(inputs=input,outputs=output)


#compile & fit
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['acc'])
es = EarlyStopping(monitor='val_acc',mode='auto',patience=200,restore_best_weights=True)
from keras.callbacks import ModelCheckpoint
mcp = ModelCheckpoint(monitor='val_loss',mode='min',save_best_only=True,
                      filepath="c:/_data/_save/MCP/fetch_covtype/K28_"+"{epoch:04d}{val_loss:.4f}.hdf5")
hist = model.fit(x_train,y_train,epochs=4096,batch_size=8192,validation_split=0.2, verbose=2,callbacks=[es,mcp])

#evaluate & predict
loss = model.evaluate(x_test,y_test,verbose=0)
y_predict = np.argmax(model.predict(x_test),axis=1)
y_test = np.argmax(y_test,axis=1)
#만약 제출하게 된다면 1부터 시작하므로 모든 데이터에 +1을 해줘야함

print(f"{r=} \nLOSS: {loss[0]} \nACC:  {accuracy_score(y_test,y_predict)}({loss[1]} by loss[1])")

plt.title('Fetch covtype Classification')
plt.xlabel('epechs')
plt.ylabel('accuracy')
plt.plot(hist.history['acc'],label='acc',color='red')
plt.plot(hist.history['val_acc'],label='val_acc',color='blue')
plt.legend()
plt.show()

# r=994
# LOSS: 0.1615818589925766
# ACC:  0.9583371580686616(0.9583371877670288 by loss[1])

# MinMaxScaler
# LOSS: 0.14452345669269562
# ACC:  0.952009133467964(0.9520091414451599 by loss[1])

# StandardScaler
# LOSS: 0.18038228154182434
# ACC:  0.9367082797870387(0.9367082715034485 by loss[1])

# MaxAbsScaler
# LOSS: 0.1562771201133728
# ACC:  0.9537704240866532(0.9537703990936279 by loss[1])

# RobustScaler
# LOSS: 0.20406576991081238
# ACC:  0.956461125390123(0.9564611315727234 by loss[1])

# r=283
# LOSS: 0.19267921149730682
# ACC:  0.9600009179364788(0.9600009322166443 by loss[1])