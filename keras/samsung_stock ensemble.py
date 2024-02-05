import numpy as np
import pandas as pd


#1. data 
df1 = pd.read_csv('C:/_data/sihum/삼성 240205.csv', index_col=0, header=0, encoding='cp949', thousands=',')
print(df1.shape) #(10296, 16)
df2 = pd.read_csv('C:/_data/sihum/아모레 240205.csv', index_col=0, header=0, encoding='cp949', thousands=',')
print(df2.shape) #(4350, 16)

# Replace empty strings with np.nan and then fill them with 0

df1.replace(r'^\s*$', np.nan, regex=True, inplace=True)
df1.fillna(0, inplace=True)

df2.replace(r'^\s*$', np.nan, regex=True, inplace=True)
df2.fillna(0, inplace=True)

df1.replace('↑', np.nan, inplace=True)
df1.fillna(0, inplace=True)

df2.replace('↑', np.nan, inplace=True)
df2.fillna(0, inplace=True)

df1.replace('↓', np.nan, inplace=True)
df1.fillna(0, inplace=True)

df2.replace('↓', np.nan, inplace=True)
df2.fillna(0, inplace=True)

df1.replace('▲', np.nan, inplace=True)
df1.fillna(0, inplace=True)

df2.replace('▲', np.nan, inplace=True)
df2.fillna(0, inplace=True)


df1.replace('▼', np.nan, inplace=True)
df1.fillna(0, inplace=True)

df2.replace('▼', np.nan, inplace=True)
df2.fillna(0, inplace=True)

for i in range(len(df1.index)):
    # df1.iloc[i, 0] = df1.iloc[i, 0].replace('.','-')
    df1.iloc[:, 0] = df1.iloc[:, 0].astype(str).str.replace('.', '-')
for i in range(len(df2.index)):
    for j in range(len(df2.iloc[i])):
        value = df2.iloc[i, j]



df1 = df1.sort_values(['일자'], ascending=[True])
df2 = df2.sort_values(['일자'], ascending=[True])
print(df1)
print(df2)

# Make sure df1 and df2 have the same number of rows
min_rows = min(df1.shape[0], df2.shape[0])
df1 = df1.iloc[:min_rows]
df2 = df2.iloc[:min_rows]


df1 = df1.values
df2 = df2.values
print(type(df1), type(df2)) #<class 'numpy.ndarray'> <class 'numpy.ndarray'>
print(df1.shape, df2.shape) #(10296, 16) (4350, 16)


def split_xy(dataset, time_steps, y_column):
    x, y = list(), list()
    for i in range(len(dataset)):
        x_end_number = i + time_steps
        y_end_number = x_end_number + y_column
        if y_end_number > len(dataset):
            break
        tmp_x = dataset[i:x_end_number, :]
        tmp_y = dataset[x_end_number:y_end_number, 3]
        x.append(tmp_x)
        y.append(tmp_y)
    return np.array(x), np.array(y)
x1,y1 = split_xy(df1, 5, 1)
x2,y2 = split_xy(df2, 5, 1)
print(x2[0,:], "\n", y2[0])
print(x2.shape, y2.shape) #(4345, 5, 16) (4345, 1)


    

from sklearn.model_selection import train_test_split
x1_train, x1_test, y1_train, y1_test = train_test_split(x1, y1, test_size=0.3, shuffle=False, random_state=777)
x2_train, x2_test, y2_train, y2_test = train_test_split(x2, y2, test_size=0.3, shuffle=False, random_state=777)
print(x2_train.shape, x2_test.shape) #(3041, 5, 16) (1304, 5, 16)
print(y2_train.shape, y2_test.shape) #(3041, 1) (1304, 1)

x1_train = np.reshape(x1_train, (x1_train.shape[0], x1_train.shape[1]*x1_train.shape[2]))
x1_test = np.reshape(x1_test, (x1_test.shape[0], x1_test.shape[1]*x1_test.shape[2]))
x2_train = np.reshape(x2_train, (x2_train.shape[0], x2_train.shape[1]*x2_train.shape[2]))
x2_test = np.reshape(x2_test, (x2_test.shape[0], x2_test.shape[1]*x2_test.shape[2]))
print(x2_train.shape, x2_test.shape) #(3041, 80) (1304, 80)



from sklearn.preprocessing import MaxAbsScaler
scaler1= MaxAbsScaler()
scaler1.fit(x1_train)
x1_train_scaled = scaler1.transform(x1_train)
x1_test_scaled = scaler1.transform(x1_test)
scaler2= MaxAbsScaler()
scaler2.fit(x2_train)
x2_train_scaled = scaler2.transform(x2_train)
x2_test_scaled = scaler2.transform(x2_test)
print(x2_train_scaled[0, :])

from keras.models import Model
from keras.layers import Dense, Input, Dropout



input1 = Input(shape=(80,))
dense1 = Dense(80, activation='relu')(input1)
dense1 = Dropout(0.5)(dense1)
dense1 = Dense(160, activation='relu')(dense1)
dense1 = Dropout(0.5)(dense1)
dense1 = Dense(320, activation='relu')(dense1)
output1 = Dense(1)(dense1)

input2 = Input(shape=(80,))
dense2 = Dense(160, activation='relu')(input2)
dense2 = Dense(320, activation='relu')(dense2)
dense2 = Dropout(0.5)(dense2)
dense2 = Dense(640, activation='relu')(dense2)
dense2 = Dropout(0.5)(dense2)
dense2 = Dense(1280, activation='relu')(dense2)
output2 = Dense(1)(dense2)

from keras.layers import concatenate
merge = concatenate([output1, output2])
output3 = Dense(1)(merge)
output4 = Dense(1)(merge)

model = Model(inputs=[input1, input2], outputs=[output3,output4])
model.summary()



from keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='loss', patience=100, mode='auto',restore_best_weights=True)
from keras.callbacks import ModelCheckpoint

# Create a ModelCheckpoint
mc = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', save_y1_predbest_only=True)

model.compile(loss='mae', optimizer='adam', metrics=['mae'])

# Convert y1_train and y2_train to float32
y1_train = y1_train.astype('float32')
y2_train = y2_train.astype('float32')


# Add the ModelCheckpoint to the callbacks list
model.fit([x1_train_scaled, x2_train_scaled], [y1_train,y2_train], epochs=10000, batch_size=32, validation_split=0.3, callbacks=[es, mc])

# Convert y1_test to float32

y1_test = y1_test.astype('float32')
y1_test = np.squeeze(y1_test, axis=1)
total_loss, loss1, loss2, mae1, mae2 = model.evaluate([x1_test_scaled, x2_test_scaled], y1_test, batch_size=1)
# total_loss, loss1, loss2, mae = model.evaluate([x1_test_scaled, x2_test_scaled], y1_test, batch_size=1)
# lose, mae = model.evaluate([x1_test_scaled, x2_test_scaled], y1_test, batch_size=1)
print('lose:', loss1,loss2)
print('mae:', mae1, mae2)

y1_pred = model.predict([x1_test_scaled, x2_test_scaled])
for i in range(5):
    print('시가:', y1_test[i], '/예측가:', y1_pred[0][i])

for i in range(5):
    print('종가:', y2_test[i], '/예측가:', y1_pred[1][i])

# y2_pred = model.predict([x1_test_scaled, x2_test_scaled])
# for i in range(5):
#     print('종가:', y2_test[i], '/예측가:', y2_pred[i])

# Load the saved model
from keras.models import load_model
saved_model = load_model('best_model.h5')

# Generate predictions on the test data
y1_pred = saved_model.predict([x1_test_scaled, x2_test_scaled])

# Create a DataFrame for the submission
submission = pd.DataFrame(y1_pred, columns=['Predicted'])

# Save the DataFrame as a CSV file
submission.to_csv("C:/_data/sihum/predicted.csv",index=False)

# 시가: [106000] /예측가: [12502.601]
# 시가: [111500] /예측가: [12448.348]
# 시가: [109000] /예측가: [12933.169]
# 시가: [110500] /예측가: [12807.637]
# 시가: [113000] /예측가: [11475.854]
# 41/41 [==============================] - 0s 772us/step
# 종가: ['191,500'] /예측가: [12502.601]
# 종가: ['183,500'] /예측가: [12448.348]
# 종가: ['160,000'] /예측가: [12933.169]
# 종가: ['156,500'] /예측가: [12807.637]
# 종가: ['153,000'] /예측가: [11475.854]