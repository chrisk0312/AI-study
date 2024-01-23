import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Dropout, Input
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Conv2D, Flatten 

train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip= True, #수평으로 뒤집기
    vertical_flip= True, #수직으로 뒤집기
    width_shift_range=0.1, #평행이동,10%
    height_shift_range=0.1, # 평행이동
    rotation_range=5, #정해진 각도만큼 이미지를 회전
    zoom_range=1.2, #축소 또는 확대
    shear_range=0.7, #좌표 하나를 고정시키고 
                    #다른 몇개의 좌표를 이동시키는 변환
    fill_mode='nearest', # 빈자리를 가장 비슷한 색으로 채움
    
)
test_datagen = ImageDataGenerator(
    rescale=1./255
)

path_train ='c:/_data/image/brain/train/'
path_test ='c:/_data/image/brain/test/'

xy_train=train_datagen.flow_from_directory(
    path_train, 
    target_size=(200,200),
    batch_size=160, #160이상을 쓰면 x통데이터로 가져올수 있다
    class_mode= 'binary',
    shuffle= True
) 
print(xy_train)
#<keras.preprocessing.image.DirectoryIterator object at 0x0000017210264520>

xy_test=test_datagen.flow_from_directory(
    path_test, 
    target_size=(200,200),
    batch_size=120,
    class_mode= 'binary',
)
print(xy_test)
print(xy_train.next())
print(xy_train[0])
# print(xy_train[16])# error! 전체 데이터/batch_size= 160/10=16개인데
                # [16]는 17번째의 값을 빼라고 하니 에러가 난다
print(xy_train[0][0]) #첫번째 배치의 x
print(xy_train[0][1]) #첫번째 배치의y
print(xy_train[0][0].shape) #(10, 200, 200, 3)

print(type(xy_train)) #<class 'keras.preprocessing.image.DirectoryIterator'>
print(type(xy_train[0])) #<class 'tuple'>
print(type(xy_train[0][0])) #<class 'numpy.ndarray'>
print(type(xy_train[0][1])) #<class 'numpy.ndarray'>

#1

x_train = xy_train[0][0]
y_train = xy_train[0][1]
x_test = xy_test[0][0]
y_test = xy_test[0][1]

print(np.unique(y_test)) #[0. 1.]
print(np.unique(np.array,return_counts=True))
print(np.unique(y_test, return_counts=True))

#2. 모델구성
input= Input(shape=(200, 200, 3))
model = Sequential()
model.add(Conv2D(9,(2,2), input_shape =(200, 200, 3)))     
model.add(Conv2D(filters=10, kernel_size=(3,3)))
model.add(Conv2D(15,(4,4)))
model.add(Flatten())
model.add(Dense(units=8))
model.add(Dense(7, input_shape= (8,)))
#                   shape=(batch_size,input_dim)
model.add(Dense(6))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='mse', optimizer='adam',metrics=['mse'])
es = EarlyStopping(monitor= 'val_loss', mode ='min', patience=20, verbose=1, restore_best_weights=True)
mcp = ModelCheckpoint(monitor='val_loss', mode='min', save_best_only=True)
model.fit(x_train,y_train,epochs=1024, batch_size=16,validation_split=0.3, verbose=2, callbacks=[es,mcp])

#evaluate& predict
loss= model.evaluate(x_test,y_test, verbose=2)
y_predict= model.predict(x_test,verbose=0)