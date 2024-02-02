import sys
import tensorflow as tf
print('텐서플로 버전 :', tf.__version__) #텐서플로 버전 : 2.9.0
print('파이썬 버전 :', sys.version) #파이썬 버전 : 3.9.18
import matplotlib.pyplot as plt
import numpy as np

from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import load_img #이미지 땡겨옴
from tensorflow.keras.preprocessing.image import img_to_array#이미지 수치화

path = "c:\_data\image\cat_and_dog\\train\Cat\\1.jpg"
img = load_img(path,
                 target_size=(150,150)
                 )
print(img)
#<PIL.Image.Image image mode=RGB size=150x150 at 0x1E571B86310>
print(type(img))
# plt.imshow(img)
# plt.show()

arr= img_to_array(img)
print(arr)
print(arr.shape) # (281, 300, 3) >>>#(150, 150, 3)
print(type(arr)) # <class 'numpy.ndarray'>

#차원증가
img = np.expand_dims(arr, axis=0)
print(img.shape) #(1, 150, 150, 3)

################################# from here expend#############
datagen =ImageDataGenerator(
    # horizontal_flip=True,
    # vertical_flip= True,
    # width_shift_range=0.2,
    # height_shift_range=0.5,
    # zoom_range=0.5,
    shear_range=180,
    fill_mode='nearest',
)

it = datagen.flow(img,
                  batch_size=1,
                  )

fig,ax = plt.subplots(nrows=1, ncols=5, figsize=(10,10))

for i in range(5):
    batch = it.next()
    print(batch)
    print(batch.shape) #(1, 150, 150, 3)
    image=batch[0].astype('uint8')
    print(image.shape) #(150, 150, 3)
    ax[i].imshow(image)
    ax[i].axis('off')
print(np.min(batch), np.max(batch))
plt.show()    

# The flow() function is a method of the ImageDataGenerator class in Keras. 
# It takes data & labels (numpy arrays) as input, and generates batches of augmented/normalized data.
# When you call flow(), it returns an Iterator that yields tuples of (x, y)
# where x is an array of image data and y is an array of corresponding labels.

     
# from keras.preprocessing.image import ImageDataGenerator

# # create a data generator
# datagen = ImageDataGenerator()

# # load image data
# x_train, y_train = load_data()

# # fit parameters from data
# datagen.fit(x_train)

# # configure batch size and retrieve one batch of images
# for X_batch, y_batch in datagen.flow(x_train, y_train, batch_size=9):
#     # create a grid of 3x3 images
#     for i in range(0, 9):
#         pyplot.subplot(330 + 1 + i)
#         pyplot.imshow(X_batch[i].reshape(28, 28), cmap=pyplot.get_cmap('gray'))
#     # show the plot
#     pyplot.show()
#     break

