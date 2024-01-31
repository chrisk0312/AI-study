import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM, Conv1D, Flatten
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score

# Load data
data = pd.read_csv('c:\\_data\\kaggle\\jena\\jena_climate_2009_2016.csv', index_col=0)

# Function to split data into input features (x) and target variable (y)
def split_xy(data, time_steps, y_column_index):
    x, y = list(), list()
    for i in range(len(data) - time_steps):
        x.append(data[i : i + time_steps])
        y.append(data[i + time_steps, y_column_index])
    return np.array(x), np.array(y)



# Normalize data
scaler = MinMaxScaler(feature_range=(0, 1))
data_normalized = scaler.fit_transform(data)

# Prepare data
time_steps = 4
y_column_index = data.columns.get_loc('T (degC)')  # Get the index of 'T (degC)' column
x, y = split_xy(data_normalized, time_steps, y_column_index)


x_train, y_train = x, y
x_train, y_train = x[:720], y[:720]


# Model
model = Sequential()
model.add(Conv1D(filters=10, kernel_size=2, input_shape=(time_steps, 14), activation='relu'))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(56, activation='relu'))
model.add(Dense(28, activation='relu'))
model.add(Dense(1))

# Compile & fit
model.compile(loss='mse', optimizer='adam', metrics=['acc'])
es = EarlyStopping(monitor='loss', mode='min', patience=300, verbose=1, restore_best_weights=True)
model.fit(x_train, y_train, epochs=1000, batch_size=800, validation_split=0.1, verbose=2, callbacks=[es])

# Predict on the next 144 rows
x_predict, y_true = x[-144:], y[-144:]
y_predict = model.predict(x_predict)

# Evaluate the model
loss = model.evaluate(x_predict, y_true)
r2 = r2_score(y_true, y_predict)

print('Loss:', loss[0])
print('R2:', r2)
y_submit = model.predict(y_column_index)
print(y_submit.shape)

# Loss: 0.00020012287131976336
# R2: 0.9332339966735401