# save_best_only
# restore_best_weight
# 에 대한 고찰
from keras.models import Sequential, load_model
from keras.layers import Dense
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.datasets import load_breast_cancer    
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

datasets = load_breast_cancer()
x = datasets.data
y = datasets.target

r = int(np.random.uniform(1,1000))
r = 88
x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.8,random_state=r)

from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler
scaler = StandardScaler().fit(x_train)  #정규분포로 바꿔줘서 스케일링

x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#model
model = Sequential()
model.add(Dense(32,input_dim=30,activation='relu'))
model.add(Dense(16,activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(1, activation='sigmoid'))

from keras.optimizers import Adam
LEARNING_RATE = 0.0001

model.compile(loss='binary_crossentropy',optimizer=Adam(learning_rate=LEARNING_RATE),metrics=['mae'])
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
es = EarlyStopping(monitor='val_loss',mode='min',patience=200,verbose=1,restore_best_weights=True)
mcp = ModelCheckpoint(monitor='val_loss',mode='min',verbose=1,save_best_only=True,
                      filepath=f"../_data/_save/MCP/k25/_"+"{epoch:04d}-{loss:.4f}.hdf5")
rlr = ReduceLROnPlateau(monitor='val_loss',mode='auto',patience=10,verbose=1,factor=0.5)

hist = model.fit(x_train,y_train,epochs=200,batch_size=10,validation_split=0.3,verbose=2,callbacks=[es,mcp,rlr])
model.save("../_data/_save/keras25_3_save_model.h5")  #가중치와 모델 모두 담겨있다


#evaluate & predict
print("============ 1. 기본출력 ============")
loss = model.evaluate(x_test,y_test,verbose=0)
y_predict = model.predict(x_test,verbose=0)

from sklearn.metrics import accuracy_score
acc = accuracy_score(y_test,np.around(y_predict))

def RMSE(y_test,y_predict):
    return np.sqrt(mean_squared_error(y_test,y_predict))

print(f"{LEARNING_RATE=}\n{loss=}\n{acc=}\nRMSE: {RMSE(y_test,y_predict)}")

# mcp출력(마지막으로 갱신된 val_loss로)
# r=88
# loss=[6.302283763885498, 1.876548171043396]
# r2=0.9146131446215406
# RMSE: 2.510434951014859

# LEARNING_RATE=1.0
# loss=[0.6428610682487488, 0.454606294631958]
# acc=0.6578947368421053
# RMSE: 0.4746255574218749

# LEARNING_RATE=0.1
# loss=[0.028034009039402008, 0.024917885661125183]
# acc=1.0
# RMSE: 0.06557261202155597

# LEARNING_RATE=0.01
# loss=[0.03021940402686596, 0.010455191135406494]
# acc=0.9912280701754386
# RMSE: 0.0908848418562058

# LEARNING_RATE=0.001
# loss=[0.018319204449653625, 0.013239954598248005]
# acc=0.9912280701754386
# RMSE: 0.07587669545439645

# LEARNING_RATE=0.0001
# loss=[0.03970786929130554, 0.03446359559893608]
# acc=0.9912280701754386
# RMSE: 0.08693296817313788

''' ============== epo 200 ============== '''
# LEARNING_RATE=1.0
# loss=[0.4831315875053406, 0.33283376693725586]
# acc=0.7894736842105263
# RMSE: 0.4004029380340427

# LEARNING_RATE=0.1
# loss=[0.6472576856613159, 0.4651584327220917]
# acc=0.6578947368421053
# RMSE: 0.47679294270653116

# LEARNING_RATE=0.01
# loss=[0.00021537212887778878, 0.00021468797058332711]
# acc=1.0
# RMSE: 0.0011644624307096108

# LEARNING_RATE=0.001
# loss=[0.0043156323954463005, 0.004010642413049936]
# acc=1.0
# RMSE: 0.022784418220860202

# LEARNING_RATE=0.0001
# loss=[0.006770640145987272, 0.006636571604758501]
# acc=1.0
# RMSE: 0.016088016013552493

''' ============== epo 500 ============== '''
# LEARNING_RATE=1.0
# loss=[0.6461811661720276, 0.46335482597351074]
# acc=0.6578947368421053
# RMSE: 0.47625708116855786

# LEARNING_RATE=0.1
# loss=[0.1141560971736908, 0.06300435215234756]
# acc=0.9736842105263158
# RMSE: 0.15985813174691274

# LEARNING_RATE=0.01
# loss=[0.0017280906904488802, 0.0016771912341937423]
# acc=1.0
# RMSE: 0.009859413742774714

# LEARNING_RATE=0.001
# loss=[0.014369217678904533, 0.011345700360834599]
# acc=0.9912280701754386
# RMSE: 0.06430882229657027

# LEARNING_RATE=0.0001
# loss=[0.028338998556137085, 0.021504422649741173]
# acc=0.9912280701754386
# RMSE: 0.0899612220574977