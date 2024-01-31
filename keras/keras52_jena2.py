import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import time
from keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score

# Function to split data into input features (x) and target variable (y)
def split_xy(data, time_steps, y_column):
    x, y = list(), list()
    for i in range(len(data) - time_steps):
        x.append(data.iloc[i : i + time_steps].values)
        y.append(data.iloc[i + time_steps][y_column])
    return np.array(x), np.array(y)

# Load data
data = pd.read_csv('c:\\_data\\kaggle\\jena\\jena_climate_2009_2016.csv', index_col=0)

#normalize data
scaler = MinMaxScaler(feature_range=(0,1))
data_normalized = scaler.fit_transform(data)

# Prepare data
time_steps = 4
x, y = split_xy(data, time_steps, 'T (degC)')

# Select training and testing data
train_size = 720
test_size = 144
x_train, y_train = x[:train_size], y[:train_size]
x_test, y_test = x[train_size:train_size + test_size], y[train_size:train_size + test_size]

# Model
model = Sequential()
model.add(LSTM(256, input_shape=(time_steps, 14), activation='relu', recurrent_dropout=0.2))
model.add(Dense(128, activation='relu'))
model.add(Dense(56, activation='relu'))
model.add(Dense(28, activation='relu'))
model.add(Dense(1))

# Compile & fit
start_time = time.time()
model.compile(loss='mse', optimizer='adam', metrics=['acc'])
es = EarlyStopping(monitor='loss', mode='min', patience=300, verbose=1, restore_best_weights=True)
model.fit(x_train, y_train, epochs=1000, batch_size=48, validation_split=0.7, verbose=2, callbacks=[es])

# Evaluate the model
loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)



r2 = r2_score(y_test, y_predict)

print('Loss:', loss[0])
print('R2:', r2)

# Loss: 13.526684761047363
# R2: -1.0528413828810166