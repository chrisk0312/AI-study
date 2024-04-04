import tensorflow as tf
from tensorflow.keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print(x_train.shape, y_train.shape, x_test.shape, y_test.shape) # (60000, 28, 28) (60000, 10) (10000, 28, 28) (10000, 10)

x_train = x_train.reshape(60000,28*28).astype('float32')/255. 
x_test = x_test.reshape(10000,28*28).astype('float32')/255.

print(x_train.shape, y_train.shape, x_test.shape, y_test.shape) # (60000, 784) (60000, 10) (10000, 784) (10000, 10)

learning_rate = 0.001

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Create a Sequential model
model = Sequential()

# Add the first Dense layer with 256 nodes, 'relu' activation function and input shape
model.add(Dense(256, activation='relu', input_shape=(784,)))

# Add the second Dense layer with 128 nodes and 'relu' activation function
model.add(Dense(128, activation='relu'))

# Add the output Dense layer with 10 nodes (for 10 classes) and 'softmax' activation function
model.add(Dense(10, activation='softmax'))

# Compile the model with 'categorical_crossentropy' loss, 'adam' optimizer and 'accuracy' metric
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model for 10 epochs
model.fit(x_train, y_train, epochs=10, batch_size=32)

# Evaluate the model on the test data
loss, accuracy = model.evaluate(x_test, y_test)

print('Test loss:', loss)
print('Test accuracy:', accuracy)