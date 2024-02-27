from keras.datasets import cifar100
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, AveragePooling2D, BatchNormalization
from keras.utils import to_categorical

#acc = 0.4 이상


#1. 데이터
(x_train, y_train), (x_test, y_test) = cifar100.load_data()

#print(x_train.shape, y_train.shape) #(50000, 32, 32, 3) (50000, 1)
#print(x_test.shape, y_test.shape) #(10000, 32, 32, 3) (10000, 1)
#print(np.unique(y_train, return_counts= True))
# [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
#        17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,
#        34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
#        51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67,
#        68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84,
#        85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99]), array([500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500,
#        500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500,
#        500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500,
#        500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500,
#        500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500,
#        500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500,
#        500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500,
#        500, 500, 500, 500, 500, 500, 500, 500, 500]



x_train = x_train.astype('float32')/255
x_test = x_test.astype('float32')/255





y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

(x_train, x_valid) = x_train[5000:], x_train[:5000]
(y_train, y_valid) = y_train[5000:], y_train[:5000]

from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
from sklearn.preprocessing import StandardScaler, RobustScaler, Normalizer

mms = MinMaxScaler(feature_range=(0,1))
#mms = StandardScaler()
#mms = MaxAbsScaler()
#mms = RobustScaler()

x_train = mms.fit_transform(x_train.reshape(-1, x_train.shape[-1])).reshape(x_train.shape)
x_test = mms.transform(x_test.reshape(-1, x_test.shape[-1])).reshape(x_test.shape)
x_valid = mms.transform(x_valid.reshape(-1, x_valid.shape[-1])).reshape(x_valid.shape)


#2. 모델구성

model = Sequential()
model.add(Conv2D(64, (3,3), activation='relu', input_shape=(32, 32, 3),strides=2))
model.add(Conv2D(64, (3,3), activation='relu',padding='same'))
model.add(MaxPooling2D(2,2))
model.add(Dropout(0.2))

# Stack 2
model.add(Conv2D(128, (3,3), activation='relu'))
model.add(Conv2D(128, (3,3), activation='relu',padding='same'))
model.add(MaxPooling2D(2,2))
model.add(Dropout(0.2))
model.add(BatchNormalization())

# Add a classifier on top of CNN
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(100, activation='softmax'))


filepath = "C:\_data\_save\MCP\_k31"

from keras.callbacks import EarlyStopping,ModelCheckpoint
import time

#x_train = x_train.reshape ( (x_train.shape[0], 32, 32, 3))
#x_test = x_test.reshape ( (x_test.shape[0], 32, 32, 3))

#3. 컴파일, 훈련


model.compile( loss= 'categorical_crossentropy', optimizer= 'adam', metrics= ['acc'])
es = EarlyStopping(monitor = 'val_loss', mode = 'auto', patience = 200, verbose = 0, restore_best_weights= True)
mcp = ModelCheckpoint(monitor='val_loss', mode = 'auto', verbose= 1, save_best_only=True, filepath= filepath)

start_time = time.time()
model.fit(x_train, y_train, batch_size=300, verbose=2, epochs= 500,validation_data=(x_valid,y_valid),shuffle=True, callbacks= [es,mcp])
end_time =time.time()

#4. 평가, 예측
results = model.evaluate(x_test,y_test)
print('loss:', results[0])
print('acc:',  results[1])
print('걸린시간 :' , end_time - start_time, "초" )

# loss: 2.2741668224334717
# acc: 0.4244000017642975
# 걸린시간 : 274.84436416625977 초

# padding/ stride
# loss: 2.111196994781494
# acc: 0.46619999408721924
# 걸린시간 : 103.90508770942688 초





