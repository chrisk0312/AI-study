from keras.datasets import fashion_mnist
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

train_datagen = ImageDataGenerator(
    rescale =1./255,
    horizontal_flip=True,
    vertical_flip=True,
    width_shift_range=0.2,
    height_shift_range=0.2,
    rotation_range=70,
    zoom_range=0.2,
    shear_range=0.7,
    fill_mode='nearest'
)

augumet_size =100

print(x_train[0].shape) #(28, 28)
# plt.imshow(x_train[0])
# plt.show()

x_data = train_datagen.flow(
    np.tile(x_train[0].reshape(28*28), augumet_size).reshape(-1, 28, 28, 1),
    np.zeros(augumet_size),
    batch_size= augumet_size,
    shuffle= False
    )

print(x_data)

# print(x_data.shape) # 투플형태라서 애러, flow에서 튜플형태로 반환됨
print(x_data[0][0].shape) #(100, 28, 28, 1)
print(x_data[0][1].shape) #(100,)
print(np.unique(x_data[0][1], return_counts=True)) #(array([0.]), array([100], dtype=int64))
print (x_data[0][0][1].shape)


plt.figure(figsize=(7,7))
for i in range(49):
    plt.subplot (7,7,i+1)
    plt.axis('off')
    plt.imshow(x_data[0][0][i],cmap='gray')
plt.show()
