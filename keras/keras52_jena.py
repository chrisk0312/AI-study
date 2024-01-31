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
data = pd.read_csv('c:\\_data\\kaggle\\jena\\jena_climate_2009_2016.csv', index_col=0)
x = data
y = data['T (degC)']
print(data.shape) #(420551, 14)
# data = np.transpose(data)
# print(data.shape) #(14, 420551)
time_steps = 4
def split_xy(data, time_steps, y_column):
    x,y = list(),list()
    for i in range(len(data) - time_steps):
        # x_end_number = i +time_steps
        # y_end_number = x_end_number + y_column
        # if y_end_number > len(data):
        #     break
        # tmp_x = data[i:x_end_number, :]
        # tmp_y = data[x_end_number: y_end_number, :]
        x.append(data[i :i+time_steps])
        y.append(data.iloc[i+time_steps][y_column])
    return np.array(x), np.array(y)



x,y = split_xy(data,time_steps,'T (degC)')
print(x, '\n', y)
print(x.shape) #(420547, 4, 14)
print(y.shape) #(420547,)


    




x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.7, shuffle= False)

# model
model = Sequential()
model.add(LSTM(256, input_shape=(4,14) , activation='relu'))
model.add(Dense(128,activation='relu'))
model.add(Dense(56,activation='relu'))
model.add(Dense(28,activation='relu'))
model.add(Dense(1))

#compile & fit
start_time = time.time()
model.compile(loss='mse',optimizer='adam',metrics=['acc'])
es = EarlyStopping(monitor='loss',mode='min',patience=30,verbose=1,restore_best_weights=True)
model.fit(x_train,y_train, epochs = 10 , batch_size= 800 , validation_split= 0.2, verbose= 2 ,callbacks=[es])

from keras.callbacks import ModelCheckpoint
end_time = time.time()



loss = model.evaluate(x_test,y_test)
y_predict = model.predict(x_test)

r2 = r2_score(y_test,y_predict)


print('loss',loss[0])
print('r2', r2)


#loss 1.949489951133728
# r2 0.9704593772050253