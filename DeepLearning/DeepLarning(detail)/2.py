from keras.models import Sequential
from keras.layers import Dense
import numpy as np

x_train = np.array([1,2,3,4,5,6,7,8,9,10])
y_train = np.array([1,2,3,4,5,6,7,8,9,10])
x_test = np.array([101,102,103,104,105,106,107,108,109,110])
y_test = np.array([101,102,103,104,105,106,107,108,109,110])

model = Sequential()
model.add(Dense(5, input_dim=1, activation='relu'))
model.add(Dense(3))
model.add(Dense(1, activation='relu'))
model.summary()


model.compile(loss='mse', optimizer='adam', metrics=['mse'])
print("model.compile")
print("model.compile")
print("model.compile")
print("model.compile")
print("model.compile")
print("model.compile")
print("model.compile")
print("model.compile")
print("model.compile")
print("model.compile")
print("model.compile")
print("model.compile")
print("model.compile")
print("model.compile")
print("model.compile")
print("model.compile")
print("model.compile")
print("model.compile")
print("model.compile")
print("model.compile")

from keras.preprocessing.image import ImageDataGenerator

# create a data generator
datagen = ImageDataGenerator(
        rotation_range=20,     # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1,      # Randomly zoom image 
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # we don't expect vertical flip in this case

# fit parameters from data
datagen.fit(x_train)

# configure batch size and retrieve one batch of images
for X_batch, y_batch in datagen.flow(x_train, y_train, batch_size=9):
    # create a grid of 3x3 images
    for i in range(0, 9):
        plt.subplot(330 + 1 + i)
        plt.imshow(X_batch[i].reshape(28, 28), cmap=plt.get_cmap('gray'))
    # show the plot
    plt.show()
    break