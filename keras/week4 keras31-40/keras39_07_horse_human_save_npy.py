from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from sklearn.model_selection import train_test_split

xy_traingen =  ImageDataGenerator(
    rescale=1./255,   
)

path_train ='c:/_data/image/horse_human/train//'

xy_train = xy_traingen.flow_from_directory(
    path_train,
    batch_size=1100,
    target_size=(100,100),
    class_mode='binary',
    color_mode='rgb',
    shuffle=True
)

x_train = xy_train[0][0] 
y_train = xy_train[0][1]

print(x_train.shape) #(1027, 100, 100, 3)

    
np_path = 'c:/_data/_save_npy/'
np.save(np_path + 'keras39_horse_human2_x_train.npy', arr=x_train)
np.save(np_path + 'keras39_horse_human2_y_train.npy', arr=y_train)
