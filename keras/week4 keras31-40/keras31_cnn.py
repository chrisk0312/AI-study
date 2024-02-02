from tensorflow.python.keras.models import Sequential
# from tensorflow.keras.models import Sequential
# from keras.models import Sequential
from tensorflow.python.keras.layers import Dense,Conv2D

model = Sequential()
#model.add(Dense(10, input_shape=(3,)))
model.add(Conv2D(10,(2,2), input_shape=(10,10,1)))
model.add(Conv2D(5,(2,2),input_shape=(9,9,10)))
model.add(Dense(5))
model.add(Dense(1))

# A Convolutional Neural Network (CNN) is a type of artificial neural network 
# that is especially effective for processing structured grid data, such as images. 
# CNNs are used for image classification, object detection, video processing, and even in natural language processing.
# They have their "neurons" arranged in three dimensions: width, height, and depth. 
# The neurons in a layer are only connected to a small region of the layer before it, called the receptive field.

