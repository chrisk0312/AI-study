import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from keras.datasets import cifar10
import tensorflow as tf
# tf.random.set_seed(777)
np.random.seed(777)
print(tf.__version__) # 2.9.0

from keras.applications import VGG16

vgg16= VGG16(weights='imagenet', 
             include_top=False, 
             input_shape=(32,32,3))
vgg16.trainable = False # Freeze the weight

model = Sequential()
model.add(vgg16)
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(10, activation='softmax'))

model.summary()

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
import time
from tensorflow.keras.callbacks import EarlyStopping

# Load CIFAR-10 data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalize pixel values to be between 0 and 1
x_train, x_test = x_train / 255.0, x_test / 255.0

# Convert class vectors to binary class matrices
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

es = EarlyStopping(monitor='val_loss', patience=10, mode='min', verbose=1)
start_time = time.time()
# Train the model
model.fit(x_train, y_train, epochs=100, batch_size=256, validation_data=(x_test, y_test))
end_time = time.time() 


# Evaluate the model
score = model.evaluate(x_test, y_test, verbose=0)
print('Time: ', end_time - start_time)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
# Time:  274.4602162837982
# Test loss: 1.203849196434021
# Test accuracy: 0.5856000185012817