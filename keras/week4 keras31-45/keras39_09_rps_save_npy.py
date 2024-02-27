from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from sklearn.model_selection import train_test_split

xy_traingen = ImageDataGenerator(
    rescale =1./255
)

path_train = 'c:/_data/image/rps/train/'

xy_train = xy_traingen.flow_from_directory(
    path_train,
    batch_size=2600,
    target_size=(100,100),
    class_mode = 'categorical',
    color_mode= 'rgb',
    shuffle=True
)

print(xy_train)

x_train = xy_train[0][0] 
y_train = xy_train[0][1]

print(x_train.shape) #(2520, 100, 100, 3)

np_path = 'c:/_data/_save_npy/'
np.save(np_path +'rps_xtrain.npy', arr=x_train)
np.save(np_path +'rps_ytrain.npy', arr=y_train)
