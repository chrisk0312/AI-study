from keras.datasets import cifar100
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, BatchNormalization
import numpy as np
import pandas as pd
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
import time
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = cifar100.load_data()
print(f"{x_train.shape=}\n{x_test.shape=}\n{y_train.shape=}\n{y_test.shape=}")
print(x_train.shape) #(50000, 32, 32, 3)
print(x_test.shape) #(10000, 32, 32, 3)
print(y_train.shape) #(50000, 1)
print(y_test.shape) #(10000, 1)
print(np.unique(y_train, return_counts=True))
# #(array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
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
#        500, 500, 500, 500, 500, 500, 500, 500, 500], dtype=int64))

x_train = x_train.astype(np.float32) / 255
x_test = x_test.astype(np.float32) / 255

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(2,2), input_shape =x_train.shape[1:], activation='relu'))
model.add(Conv2D(32,(2,2), activation= 'relu'))
model.add(Conv2D(64,(2,2), activation= 'relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.1))
model.add(Conv2D(filters=32,kernel_size=(2,2),activation='relu'))
model.add(Conv2D(64,(2,2),activation='relu'))
model.add(Conv2D(128,(2,2),activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.1))
model.add(Flatten())
model.add(Dense(512,activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(100,activation='softmax'))

#compile & fit
start_time = time.time()
model.compile(loss = 'categorical_crossentropy',optimizer='adam',metrics=['acc'])
es = EarlyStopping(monitor='val_acc',mode='auto',patience=50,restore_best_weights=True)
hist = model.fit(x_train,y_train,epochs=100,batch_size=1024,validation_data=(x_test,y_test),verbose=2,callbacks=[es])
end_time = time.time()


# evaluate & predict
loss = model.evaluate(x_test,y_test)
y_predict = model.predict(x_test)

print(f"time: {end_time-start_time}sec")
print(f"LOSS: {loss[0]}\nACC:  {loss[1]}")

model.save(f"C:/_data/_save/keras31/CNN5_cifar10_ACC{loss[1]}.h5")

plt.title("CIFAR10")
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.plot(hist.history['acc'],label='acc',color='red')
plt.plot(hist.history['val_acc'],label='val_acc',color='blue')
plt.legend()
plt.show()