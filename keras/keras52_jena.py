import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.model_selection import train_test_split
import pandas as pd
import time
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score

#data
path = pd.read_csv('c:\\_data\\kaggle\\jena\\jena_climate_2009_2016.csv')
x = path
y = path['T (degC)'] 
x_timesteps = []
y_timesteps = []
k=24
for i in range(len(x)-k):
    x_timesteps.append(x.iloc[i:i+k].values)
    y_timesteps.append(y.iloc[i+k])

X_timesteps = np.asarray(x_timesteps)
y_timesteps = np.asarray(y_timesteps)
print(X_timesteps.shape) #(420527, 24, 2)
X_timesteps=X_timesteps.reshape((X_timesteps.shape[0], -1))
print(X_timesteps.shape) #(420527, 48)

x_train, x_test, y_train, y_test = train_test_split(X_timesteps, y_timesteps, test_size=0.25, random_state=42)

# model
model = Sequential()
model.add(Dense(800, input_dim = 8, activation='relu'))

model.add(Dense(600, ))
model.add(Dense(400, ))

model.add(Dense(200, ))
model.add(Dense(128, ))
model.add(Dense(64, ))
model.add(Dense(32, ))
model.add(Dense(8, activation='relu'))
model.add(Dense(1))

#compile & fit
start_time = time.time()
model.compile(loss='mse',optimizer='adam',metrics=['mse'])
es = EarlyStopping(monitor='val_loss',mode='min',patience=30,verbose=1,restore_best_weights=True)
from keras.callbacks import ModelCheckpoint
mcp = ModelCheckpoint(monitor='val_loss',mode='min',save_best_only=True,
                      filepath="c:/_data/_save/MCP/kaggle_bike/K28_"+"{epoch:04d}{val_loss:.4f}.hdf5")
hist = model.fit(x_train,y_train,epochs=123,batch_size=32,verbose=2,validation_split=0.3)#,callbacks=[es,mcp])
end_time = time.time()
#evaluate & predict 
loss = model.evaluate(x_test,y_test,verbose=0)
y_predict = model.predict(x_test)
r2 = r2_score(y_test,y_predict)
y_submit = model.predict(y)

print(f"Time: {round(end_time-start_time,2)}sec")