from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, BatchNormalization
import numpy as np
import pandas as pd
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
import time
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# print(f"{x_train.shape=}\n{x_test.shape=}\n{y_train.shape=}\n{y_test.shape=}")
# x_train.shape=(50000, 32, 32, 3)
# x_test.shape=(10000, 32, 32, 3)
# y_train.shape=(50000, 1)
# y_test.shape=(10000, 1)
# print(np.unique(y_train, return_counts=True))
# (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), array([5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000],dtype=int64))

# acc > 0.77

x_train = x_train.astype(np.float32) / 255
x_test = x_test.astype(np.float32) / 255

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# model
model = Sequential()
model.add(Conv2D(filters=32,kernel_size=(2,2),input_shape=x_train.shape[1:],activation='relu'))
model.add(Conv2D(32,(2,2),activation='relu'))
model.add(Conv2D(64,(2,2),activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.1))
model.add(Conv2D(filters=32,kernel_size=(2,2),activation='relu'))
model.add(Conv2D(64,(2,2),activation='relu'))
model.add(Dense(10))
model.add(Conv2D(128,(2,2),activation='relu'))
model.add(MaxPooling2D())
# model.add(Dropout(0.1))
# model.add(Conv2D(filters=32,kernel_size=(2,2),activation='relu'))
# model.add(Conv2D(128,(2,2),activation='relu'))
# model.add(Conv2D(128,(2,2),activation='sigmoid'))
# model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.1))
model.add(Flatten())
model.add(Dense(512,activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(10,activation='softmax'))

# compile & fit
start_time = time.time()
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['acc'])
es = EarlyStopping(monitor='val_acc',mode='auto',patience=50,restore_best_weights=True)
hist = model.fit(x_train,y_train,epochs=1000,batch_size=1000,validation_data=(x_test,y_test),verbose=2,callbacks=[es])
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

# time: 530.7889325618744sec
# LOSS: 0.9106385707855225
# ACC:  0.7649000287055969

# 32relu,64relu,drop0.1,64relu,128relu,drop0.1,Flatten,512
# time: 366.77252554893494sec
# LOSS: 2.1206843852996826
# ACC:  0.6802999973297119