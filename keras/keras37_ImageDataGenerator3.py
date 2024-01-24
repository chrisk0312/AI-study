from keras.preprocessing.image import ImageDataGenerator
import numpy as np
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

path_train ='c:/_data/image/cat-and-dog/train/'
# path_test ='c:/_data/image/cat_and_dog/test/'

xy_train = xy_traingen.flow_from_directory(
    path_train,
    batch_size=20000,
    target_size=(100,100),
    class_mode='binary',
    color_mode='rgb',
    shuffle=True
)

# xy_test = xy_traingen.flow_from_directory(
#     path_test,
#     batch_size=1000,
#     target_size=(200,200),
#     class_mode='binary'
# )

print(len(xy_train[0][0]))
print(len(xy_train[0][1]))

x_train = xy_train[0][0]
y_train = xy_train[0][1]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test =  train_test_split(x_train, y_train, train_size= 0.8, random_state=777)

print(x_train.shape)

# x_train = x_train/255.
# x_test = x_test/255.

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
model.fit(x_train, y_train, epochs= 1000, batch_size= 10, validation_split= 0.2, callbacks=[es])
endTime = tm.time()

#평가 예측
loss = model.evaluate(x_test, y_test)
predict = np.round(model.predict(x_test))
# print(predict)
print('loss : ', loss[0])
print('acc : ', loss[1])
print('time :', np.round(endTime - startTime, 2) ,"sec")

