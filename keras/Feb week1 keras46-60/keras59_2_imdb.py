from keras.datasets import imdb
import numpy as np 
from keras.utils import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, Flatten

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=10000)

print(x_train)
print(y_train.shape, y_test.shape) # (25000,) (25000,
print(x_test.shape, y_test.shape) # (25000,) (25000,    
print(len(x_train[0]), len(x_test[0])) #218 68  
print(y_train[:20]) # [1 0 0 1 0 0 1 0 1 0 1 0 0 0 0 0 1 1 0 1]
print(np.unique(y_train, return_counts=True)) # (array([0, 1]), array([12500, 12500]))

# Pad sequences
x_train = pad_sequences(x_train, maxlen=100)
x_test = pad_sequences(x_test, maxlen=100)

# Define the model
model = Sequential()
model.add(Embedding(10000, 32, input_length=100))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_test, y_test))

# Evaluate the model
loss, accuracy = model.evaluate(x_test, y_test)
print("Test Accuracy: ", accuracy)# Test Accuracy:  0.8530399799346924