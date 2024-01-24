from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, Flatten, MaxPooling2D
from sklearn.model_selection import train_test_split 

import time as tm


 #1
np_path= 'c:/_data/_save_npy/'

x_train= np.load(np_path + 'keras39_1_x_train.npy')
y_train= np.load(np_path + 'keras39_1_y_train.npy')

print(x_train.shape, y_train.shape)#(160, 100, 100, 1) (160,)

x_train, y_train, x_test, y_test = train_test_split(x_train, y_train, test_size=0.3)


#모델구성
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, Flatten, MaxPooling2D
model = Sequential()

model.add(Conv2D(32, (2,2), input_shape=(100,100,3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(1, activation='sigmoid'))

from keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', mode = 'min', patience=300, restore_best_weights=True)

import time as tm
#컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
startTime = tm.time()
model.fit(x_train,y_train, 
          steps_per_epoch=16,
          epochs= 100, 
          batch_size= 10, 
          validation_split= 0.2, 
          callbacks=[es])
endTime = tm.time()

#평가 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss[0])
print('acc : ', loss[1])
print('time :', np.round(endTime - startTime, 2) ,"sec")