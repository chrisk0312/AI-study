import numpy as np
import pandas as pd

#1. data 
df1 = pd.read_csv('C:/_data/sihum/삼성 240205.csv', index_col=0, header=0, encoding='cp949')
print(df1.shape) #(10296, 16)
df2 = pd.read_csv('C:/_data/sihum/아모레 240205.csv', index_col=0, header=0, encoding='cp949')
print(df2.shape) #(4350, 16)

for i in range(len(df1.index)):
    df1.iloc[i, 0] = df1.iloc[i, 0].replace('.','-')
for i in range(len(df2.index)):
    for j in range(len(df2.iloc[i])):\
        value = df2.iloc[i, j]
if isinstance(value, str):
    df2.iloc[i, j] = value.replace('.','-')
    df2.iloc[i, j] = df2.iloc[i, j].replace('.','-')


df1 = df1.sort_values(['일자'], ascending=[True])
df2 = df2.sort_values(['일자'], ascending=[True])
print(df1)
print(df2)

df1 = df1.values
df2 = df2.values
print(type(df1), type(df2)) #<class 'numpy.ndarray'> <class 'numpy.ndarray'>
print(df1.shape, df2.shape) #(10296, 16) (4350, 16)

np.save('C:/_data/sihum/samsung_stock.npy', arr=df1)
np.save('C:/_data/sihum/amore_stock.npy', arr=df2)

samsung = np.load('C:/_data/sihum/samsung_stock.npy', allow_pickle=True)
amore = np.load('C:/_data/sihum/amore_stock.npy', allow_pickle=True)
print(samsung.shape) #(10296, 16)
print(amore.shape) #(4350, 16)

def split_xy(dataset, time_steps, y_column):
    x, y = list(), list()
    for i in range(len(dataset)):
        x_end_number = i + time_steps
        y_end_number = x_end_number + y_column
        if y_end_number > len(dataset):
            break
        tmp_x = dataset[i:x_end_number, :]
        tmp_y = dataset[x_end_number:y_end_number, 0]
        x.append(tmp_x)
        y.append(tmp_y)
    return np.array(x), np.array(y)
x,y = split_xy(samsung, 5, 1)
print(x[0,:], "\n", y[0])
print(x.shape, y.shape) #(10291, 5, 16) (10291, 1)


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, shuffle=False, random_state=1)
print(x_train.shape, x_test.shape) #(7203, 5, 16) (3088, 5, 16)
print(y_train.shape, y_test.shape) #(7203, 1) (3088, 1)

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1]*x_train.shape[2])
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1]*x_test.shape[2])
print(x_train.shape, x_test.shape) #(7203, 80) (3088, 80)


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

x_train = pd.DataFrame(x_train).applymap(lambda x: str(x).replace(',', '') if isinstance(x, str) else x)
x_train = x_train.apply(pd.to_numeric, errors='coerce').fillna(0).values

x_test = pd.DataFrame(x_test).applymap(lambda x: str(x).replace(',', '') if isinstance(x, str) else x)
x_test = x_test.apply(pd.to_numeric, errors='coerce').fillna(0).values

scaler = StandardScaler()
scaler.fit(x_train)
x_train_scaled = scaler.transform(x_train)
x_test_scaled = scaler.transform(x_test)
print(x_train_scaled[0, :])

from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.ensemble import VotingRegressor

model = Sequential()
model.add(Dense(80, input_dim=80, activation='relu'))
model.add(Dense(160, activation='relu'))
model.add(Dense(320, activation='relu'))
model.add(Dense(40, activation='relu'))
model.add(Dense(1))


y_train = pd.Series(y_train.flatten())
y_train = pd.to_numeric(y_train, errors='coerce').fillna(0).values
y_test = pd.Series(y_test.flatten())
y_test = pd.to_numeric(y_test, errors='coerce').fillna(0).values

from keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='loss', patience=1000, mode='auto')
model.compile(loss='mae', optimizer='adam', metrics=['mae'])
model.fit(x_train_scaled, y_train, epochs=10000, batch_size=32, validation_split=0.2, callbacks=[es])


loss, mae = model.evaluate(x_test_scaled, y_test, batch_size=1)
print('loss:', loss)
print('mae:', mae)

y_pred = model.predict(x_test_scaled)
for i in range(5):
    print('종가:', y_test[i], '/예측가:', y_pred[i])


# loss: 0.07110719382762909
# mse: 0.07110719382762909

# loss: 0.06947251409292221
# mae: 0.06947251409292221






