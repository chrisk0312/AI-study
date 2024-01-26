import numpy as np
from keras.datasets import cifar10
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
from sklearn.preprocessing import OneHotEncoder
from keras.utils import to_categorical
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint


#1 데이터
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train/255.
x_test = x_test/255.

train_datagen = ImageDataGenerator(
    horizontal_flip=True,
    vertical_flip= True,
    width_shift_range= 0.2,
    height_shift_range= 0.2,
    rotation_range=30,
    zoom_range=0.2,
    shear_range=0.2,
    fill_mode= 'nearest'
)

augment_size = 50000

randidx = np.random.randint(x_train.shape[0], size =augment_size)



x_augmented = x_train[randidx].copy()
y_augmented = y_train[randidx].copy()
print(x_augmented)
print(x_augmented.shape) #(50000, 32, 32, 3)
print(y_augmented.shape) #(50000, 1)



x_augmented = x_augmented.reshape(-1,32,32,3)

x_augmented = train_datagen.flow(
    x_augmented, y_augmented,
    batch_size= augment_size,
    shuffle =False
).next()[0]

x_train = x_train.reshape(50000,32,32,3)
x_test = x_test.reshape(10000,32,32,3)

x_train = np.concatenate((x_train, x_augmented))
y_train = np.concatenate((y_train, y_augmented))

ohe = OneHotEncoder(sparse= False)
y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)
y_train = ohe.fit_transform(y_train)
y_test = ohe.fit_transform(y_test)


(x_train, x_valid) = x_train[5000:], x_train[:5000]
(y_train, y_valid) = y_train[5000:], y_train[:5000]

#2
model = Sequential()
model.add(Conv2D(9,(2,2), input_shape = (32,32,3)))
model.add(Conv2D(filters=10,kernel_size=(3,3)))
model.add(Conv2D(15,(4,4)))
model.add(Flatten())
model.add(Dense(units=8))
model.add(Dense(7, input_shape =(8,)))
model.add(Dense(6))
model.add(Dense(10, activation= 'softmax'))


filepath = "C:\_data\_save\MCP\\"


#3. 컴파일, 훈련
model.compile ( loss = 'categorical_crossentropy', optimizer = 'adam', metrics= 'acc')
es = EarlyStopping(monitor = 'val_loss', mode = 'auto', patience = 200, verbose = 0, restore_best_weights= True)
mcp = ModelCheckpoint(monitor='val_loss', mode = 'auto', verbose= 1, save_best_only=True, filepath= filepath)
model.fit(x_train, y_train, batch_size = 512, verbose=3, epochs= 100, validation_data= (x_valid, y_valid), callbacks= [es, mcp])


#4. 평가, 예측
results = model.evaluate(x_test,y_test)
print('loss:', results[0])
print('acc:',  results[1])





