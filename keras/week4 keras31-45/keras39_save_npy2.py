from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, Flatten, MaxPooling2D


import time as tm


 #1
np_path= 'c:/_data/_save_npy/'
# np.save(np_path + 'keras39_1_x_train.npy',arr= xy_train[0][0])
# np.save(np_path + 'keras39_1_x_train.npy',arr= xy_train[0][1])
# np.save(np_path + 'keras39_1_x_test.npy',arr= xy_train[0][0])
# np.save(np_path + 'keras39_1_x_test.npy',arr= xy_train[0][1])

x_train= np.load(np_path + 'keras39_1_x_train.npy')
y_train= np.load(np_path + 'keras39_1_y_train.npy')
x_test= np.load(np_path + 'keras39_1_x_test.npy')
y_test= np.load(np_path + 'keras39_1_y_test.npy')

print(x_train.shape, y_train.shape)#(160, 100, 100, 1) (160,)
print(x_test.shape, y_test.shape)#((120, 100, 100, 1) (120,)




#2모델구성

model = Sequential()
model.add(Conv2D(32, (2,2), input_shape=(100,100,1), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(1, activation='sigmoid'))



#3 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
startTime = tm.time()
model.fit(x_train,y_train, 
                    steps_per_epoch=16, #전체데이터/batch= 160/10= 16, 17은 에러, 15는 데이터손실
                    epochs= 10, 
                    #batch_size= 32, #fit_generator에서는 에러, fit에서는 안먹힘
                   validation_split= 0.2,
                )
endTime = tm.time()

#평가 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss[0])
print('acc : ', loss[1])
print('time :', np.round(endTime - startTime, 2) ,"sec")
