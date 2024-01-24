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


path_train ='c:/_data/image/cat-and-dog/train/'
path_test ='c:/_data/image/cat_and_dog/test/'

xy_train = xy_traingen.flow_from_directory(
    path_train,
    batch_size=1000,
    target_size=(100,100),
    class_mode='binary',
    color_mode='rgb',
    shuffle=True
)

xy_test = xy_traingen.flow_from_directory(
    path_test,
    batch_size=1000,
    target_size=(100,100),
    class_mode='binary',
    color_mode='rgb',
    shuffle= True,
    )

print(xy_train)
print(xy_train[0][0].shape) #(160, 100, 100, 1)
print(xy_train[0][1].shape) #(160,)

# print(xy_test[0][0].shape) #(160, 100, 100, 1)
# print(xy_test[0][1].shape) #(160,)

for data_batch, labels_batch in xy_train:
    print("Data batch shape:", data_batch.shape)
    print("Labels batch shape:", labels_batch.shape)
    break
'''
np_path= 'c:/_data/_save_npy/'
np.save(np_path + 'keras39_3_x_train.npy',arr= xy_train[0][0])
np.save(np_path + 'keras39_3_y_train.npy',arr= xy_train[0][1])
# np.save(np_path + 'keras39_3_x_test.npy',arr= xy_test[0][0])
# np.save(np_path + 'keras39_3_y_test.npy',arr= xy_test[0][1])

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
model.fit(xy_train, 
          steps_per_epoch=16,
          epochs= 100, 
          batch_size= 10, 
          validation_split= 0.2, 
          callbacks=[es])
endTime = tm.time()

#평가 예측
loss = model.evaluate(xy_test)
print('loss : ', loss[0])
print('acc : ', loss[1])
print('time :', np.round(endTime - startTime, 2) ,"sec")
'''
