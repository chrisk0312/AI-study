import numpy as np
from keras.datasets import cifar100
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
from sklearn.preprocessing import OneHotEncoder
from keras.utils import to_categorical
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from keras.datasets import cifar100
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping,ModelCheckpoint

#1
(x_train,y_train), (x_test, y_test) = cifar100.load_data()
x_train = x_train/255.
x_test = x_test/255.

train_datagen = ImageDataGenerator(
    horizontal_flip= True,
    vertical_flip= True,
    width_shift_range=0.2,
    height_shift_range=0.2,
    rotation_range=30,
    zoom_range=0.2,
    shear_range=0.2,
    fill_mode= 'nearest'
)

augment_size= 50000

randidx = np.random.randint(x_train.shape[0], size = augment_size)

x_augmented = x_train[randidx].copy()
y_augmented = y_train[randidx].copy()
