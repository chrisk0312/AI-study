from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, Flatten, MaxPooling2D


import time as tm


 #1
xy_traingen =  ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    vertical_flip=True,
    width_shift_range= 0.1,
    height_shift_range= 0.1,
    rotation_range=5,       
    zoom_range=1.2,         
    shear_range=0.7,      
    fill_mode=''
)
xy_traingen = ImageDataGenerator(
    rescale=1./255
)
path_train ='c:/_data/image/brain/train/'
path_test ='c:/_data/image/brain/test/'
xy_train = xy_traingen.flow_from_directory(
    path_train,
    batch_size=10,
    target_size=(100,100),
    class_mode='binary',
    color_mode='grayscale',
    shuffle=True
)
xy_test = xy_traingen.flow_from_directory(
    path_test,
    batch_size=10,
    target_size=(100,100),
    class_mode='binary',
    color_mode='grayscale',
    shuffle= True,
)
print(xy_train)
print(xy_train[0][0].shape) #(160, 100, 100, 1)
print(xy_train[0][1].shape) #(160,)


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
model.fit(xy_train, 
                    steps_per_epoch=16, #전체데이터/batch= 160/10= 16, 17은 에러, 15는 데이터손실
                    epochs= 10, 
                    #batch_size= 32, #fit_generator에서는 에러, fit에서는 안먹힘
                   # validation_split= 0.2,
                   validation_data=xy_test,
)
endTime = tm.time()

#평가 예측
loss = model.evaluate(xy_test)
print('loss : ', loss[0])
print('acc : ', loss[1])
print('time :', np.round(endTime - startTime, 2) ,"sec")
