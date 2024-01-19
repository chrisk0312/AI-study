from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.datasets import load_boston    # pip install scikit-learn==1.1.3
import time
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler

import warnings
warnings.filterwarnings('ignore')

datasets = load_boston()
x = datasets.data
y = datasets.target

r2 = 0

# while r2 < 0.8:
r = int(np.random.uniform(1,1000))
r = 88
x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.8,random_state=r)


# scaler = MinMaxScaler().fit(x_train)    #최솟값을 0 최댓값을 1로 스케일링
scaler = StandardScaler().fit(x_train)  #정규분포로 바꿔줘서 스케일링
# scaler = MaxAbsScaler().fit(x_train)    #
# scaler = RobustScaler().fit(x_train)    #

x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#model

input = Input(shape=(13,))
d1 = Dense(32, activation='relu')(input)
drop1 = Dropout(0.2)(d1)
d2 = Dense(16, activation='relu')(drop1)
drop2 = Dropout(0.2)(d2)
d3 = Dense(8, activation='relu')(drop2)
output = Dense(1)(d3)

model = Model(inputs=input,outputs=output)


#compile & fit
model.compile(loss='mse',optimizer='adam',metrics=['mae'])
start_time = time.time()

from keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss',mode='min',patience=50,verbose=1,restore_best_weights=True)
mcp = ModelCheckpoint(monitor='val_loss',mode='auto',save_best_only=True,verbose=1,
                      filepath="c:/_data/_save/MCP/boston/K28_"+"{epoch:04d}{val_loss:.4f}.hdf5")
hist = model.fit(x_train,y_train,epochs=12345,batch_size=10,validation_split=0.3,verbose=2)#,callbacks=[es,mcp])

#evaluate & predict
loss = model.evaluate(x_test,y_test,verbose=0)
result = model.predict(x,verbose=0)
y_predict = model.predict(x_test,verbose=0)

r2 = r2_score(y_test,y_predict)
end_time = time.time()

def RMSE(y_test,y_predict):
    return np.sqrt(mean_squared_error(y_test,y_predict))

print(f"Time: {round(end_time-start_time,2)}sec")
print(f"{r=}\n{loss=}\n{r2=}\nRMSE: {RMSE(y_test,y_predict)}")


# CPU Time: 559.09sec
# Time: 790.97sec