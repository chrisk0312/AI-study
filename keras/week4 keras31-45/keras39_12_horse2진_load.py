from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, Flatten, MaxPooling2D
import time as tm
from keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score

np_path = 'c:/_data/_save_npy/'

x_train = np.load(np_path + 'keras39_horse_human2_x_train.npy')
y_train = np.load(np_path + 'keras39_horse_human2_y_train.npy')

x_train, x_test, y_train, y_test =  train_test_split(x_train, y_train, train_size= 0.8, random_state=777)

print(x_train.shape) #(821, 100, 100, 3)

model = Sequential()
model.add(Conv2D(32, (2,2), input_shape = (100,100,3), activation= 'relu'))
model.add(MaxPooling2D())
model.add(Conv2D(64,(3,3), activation='relu'))
model.add(MaxPooling2D())
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(1,activation='sigmoid'))
model.summary()


model.compile(loss='binary_crossentropy', optimizer='adam', metrics= ['acc'])
es =  EarlyStopping(monitor = 'val_loss', mode = 'min', patience=300, restore_best_weights=True)
model.fit(x_train,y_train,
          epochs= 10,
          batch_size=10,
          validation_split=0.2,
          callbacks=[es],
          )

loss= model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)
print(y_predict.shape)


print('loss :',loss[0])
print('acc :',loss[1])




