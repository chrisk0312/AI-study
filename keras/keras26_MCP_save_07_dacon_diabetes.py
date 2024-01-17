from sklearn.datasets import load_diabetes
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
import time

#1. 데이터
datasets = load_diabetes()
x = np.array(datasets.data)
y = np.array(datasets.target)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.25, shuffle= True, random_state = 442)


from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
from sklearn.preprocessing import StandardScaler, RobustScaler

#mms = MinMaxScaler()
mms = StandardScaler()
#mms = MaxAbsScaler()
#mms = RobustScaler()

mms.fit(x_train)
x_train= mms.transform(x_train)
x_test= mms.transform(x_test)

#2. 모델구성
model = Sequential()
model.add(Dense(5, input_dim = 10))
model.add(Dense(12))
model.add(Dense(24))
model.add(Dense(48))
model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(20))
model.add(Dense(8))
model.add(Dense(6))
model.add(Dense(4))
model.add(Dense(2))
model.add(Dense(1))



# #3. 컴파일, 훈련
from keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor = 'val_loss', mode = 'min', patience = 10, verbose = 2, restore_best_weights= True)
mcp = ModelCheckpoint(monitor='val_loss', mode = 'auto', verbose= 1, save_best_only=True, filepath='../_data/_save/MCP/keras26_07_MCP.hdf5')

model.compile(loss= 'mse', optimizer= 'adam', metrics= 'acc' ) #mae 2.64084 r2 0.8278   mse 12.8935 r2 0.82
hist = model.fit(x_train, y_train, callbacks=[es,mcp], epochs= 500, batch_size = 5, validation_split= 0.3)


#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print("로스 :", loss)
y_predict = model.predict(x_test)
result = model.predict(x)
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print("R2 스코어 :", r2)
# print("걸린 시간 :", round(end_time - start_time, 2), "초")
# import matplotlib.pyplot as plt
# plt.rcParams['font.family'] = 'Malgun Gothic'
# plt.rcParams['axes.unicode_minus']= False
# plt.figure(figsize= (9,6))
# plt.plot(hist.history['loss'], c = 'red', label = 'loss', marker = '.')
# plt.plot(hist.history['val_loss'], c = 'blue', label = 'val_loss', marker = '.')
# plt.legend(loc = 'upper right')
# plt.title("당뇨병 LOSS")
# plt.xlabel('epoch')
# plt.ylabel('loss')
# plt.grid()
# plt.show()
#False
#로스 : 2513.03662109375
#R2 스코어 : 0.5959205821049586
#True
#로스 : 2686.110107421875
#R2 스코어 : 0.568091475683709
#metrics
#로스 : 2447.14013671875
#R2 스코어 : 0.6261307996541818


#mms = MinMaxScaler()
#로스 : [2493.5859375, 0.0]
#R2 스코어 : 0.6190349078359285


#mms = StandardScaler()
#로스 : [2405.402099609375, 0.0]
#R2 스코어 : 0.6325074197149114


#mms = MaxAbsScaler()
#로스 : [2603.160400390625, 0.0]
#R2 스코어 : 0.6022943290304734

#mms = RobustScaler()
#로스 : [2751.904052734375, 0.0]
#R2 스코어 : 0.5795696017777313

#R2 스코어 : 0.6368696550774082