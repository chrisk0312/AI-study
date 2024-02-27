from sklearn.datasets import fetch_california_housing
from keras.models import Model
from keras.layers import Dense, Dropout, Input 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import time
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping

#data
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target
print(x.shape,y.shape,sep='\n')
print(datasets.feature_names)   
#['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']
r2=0
# while r2 < 0.6: 
r = int(np.random.uniform(1,1000))
r = 176
# r = 130
x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.7,random_state=r)

from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler
# scaler = MinMaxScaler().fit(x_train)    #최솟값을 0 최댓값을 1로 스케일링
# scaler = StandardScaler().fit(x_train)  #정규분포로 바꿔줘서 스케일링
# scaler = MaxAbsScaler().fit(x_train)    #
scaler = RobustScaler().fit(x_train)    #

x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#model
# model = Sequential()
# model.add(Dense(32,input_dim=8,activation='relu'))
# model.add(Dense(64,activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(32,activation='relu'))
# model.add(Dense(16,activation='relu'))
# model.add(Dropout(0.1))
# model.add(Dense(8,activation='relu'))
# model.add(Dense(1))

#함수형

input= Input(shape=(8,)) 
dense1 = Dense(32)(input)
dense2 = Dense(64)(dense1)
output= Dense(1)(dense2)
model = Model(inputs=input, outputs= output)






#compile fit
model.compile(loss='mse',optimizer='adam',metrics=['mse'])
start_time = time.time()
es = EarlyStopping(monitor='val_loss',mode='min',patience=20,restore_best_weights=True,verbose=1)
from keras.callbacks import ModelCheckpoint
mcp = ModelCheckpoint(monitor='val_loss',mode='min',save_best_only=True,
                      filepath="c:/_data/_save/MCP/california/k28_"+"{epoch:04d}{val_loss:.4f}.hdf5")
hist = model.fit(x_train,y_train,epochs=1024,batch_size=64,validation_split=0.3,verbose=2,callbacks=[es,mcp])

#evaluate predict
loss = model.evaluate(x_test,y_test,verbose=0)
result = model.predict(x,verbose=0)
y_predict = model.predict(x_test,verbose=0)

r2 = r2_score(y_test,y_predict)
end_time = time.time()
print(f"Time: {round(end_time-start_time,2)}sec")
print(f"{r=}\n{loss=}\n{r2=}")

plt.figure(figsize=(9,6))
plt.plot(hist.history['loss'],color='red',label='loss',marker='.')
plt.plot(hist.history['val_loss'],color='blue',label='val_loss',marker='.')
# plt.plot(range(128),np.array([hist.history['loss'],hist.history['val_loss']]).T,label=['loss','val_loss'])
plt.legend(loc='upper right')
plt.title('california loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.grid()
plt.show()

# Epoch 236: early stopping
# Time: 35.14sec
# r=176
# loss=0.45041927695274353
# r2=0.661553367308052

# MinMaxScaler
# loss=[0.27136003971099854, 0.27136003971099854]
# r2=0.796099072829003

# StandardScaler
# loss=[0.26987117528915405, 0.26987117528915405]
# r2=0.7972178874447766

# MaxAbsScaler
# loss=[0.3393670916557312, 0.3393670916557312]
# r2=0.7449983684198338

# RobustScaler
# loss=[0.293599396944046, 0.293599396944046]
# r2=0.7793883660319727

# dropout
# r=176
# loss=[0.2723971903324127, 0.2723971903324127]       
# r2=0.7953197636967315