from keras.datasets import fashion_mnist
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train = x_train/255.
x_test = x_test/255.


train_datagen = ImageDataGenerator(
  #  rescale =1./255,
    horizontal_flip=True,
    vertical_flip=True,
    width_shift_range=0.2,
    height_shift_range=0.2,
    rotation_range=70,
    zoom_range=0.2,
    shear_range=0.7,
    fill_mode='nearest'
)

augumet_size =40000

randidx = np.random.randint(x_train.shape[0], size=augumet_size)
            #np.random.randit(60000,40000)
print(randidx) #[13412 23022 31663 ...  3101 10930 50679]
print(np.min(randidx),np.max(randidx)) #4 59998

x_augumented = x_train[randidx].copy()
y_augumented = y_train[randidx].copy()

# print(x_augumented)
# print(x_augumented.shape) #(40000, 28, 28)
# print(y_augumented)
# print(y_augumented.shape) #(40000,)

x_augumented = x_augumented.reshape(
    x_augumented.shape[0],
    x_augumented.shape[1],
    x_augumented.shape[2],1)

x_augumented = train_datagen.flow(
    x_augumented, y_augumented,
    batch_size=augumet_size,
    shuffle= False
).next()[0]

print(x_augumented)
print(x_augumented.shape) #(40000, 28, 28, 1)

print(x_train.shape)
x_train = x_train.reshape(60000,28,28,1)
x_test = x_test.reshape(10000,28,28,1)

print(x_train.shape, x_augumented.shape)

x_train = np.concatenate((x_train, x_augumented))
y_train = np.concatenate((y_train, y_augumented)) #
print(x_train.shape, y_train.shape) #(100000, 28, 28, 1) (100000,)








