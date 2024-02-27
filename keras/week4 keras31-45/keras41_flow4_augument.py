from keras.datasets import fashion_mnist
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()


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

# In the context of your code, "augument" seems to be a misspelling of "augment".
# Data augmentation is a strategy used in machine learning to increase the diversity and 
# amount of training data through random transformations and modifications.
# In the context of image data, this can involve transformations like rotation, rescaling, horizontal 
# or vertical flip, zooming, shearing, and shifting height or width.
# In your code, augumet_size = 40000 is defining the number of augmented images to generate. 
# The ImageDataGenerator is then used to generate these augmented images.
# The x_augumented and y_augumented variables are copies of the images and 
# labels in your training data that will be used for augmentation. 
# The flow() method of ImageDataGenerator is then used to generate the augmented images.
# The augmented images are then concatenated with the original training data.
# The purpose of data augmentation is to increase the diversity of the training data and
# improve the model's ability to generalize to new, unseen data.

# So, in summary, "augument" in your code refers to data augmentation, 
# a technique used to generate more training data and improve the model's ability to generalize.







