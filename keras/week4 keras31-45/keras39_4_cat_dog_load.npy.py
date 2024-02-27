import numpy as np
import time
from keras.callbacks import EarlyStopping
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, Flatten, Dropout,MaxPooling2D, GlobalAveragePooling2D, Input
from sklearn.model_selection import train_test_split
import pandas as pd
import datetime
import os
#1
np_path = 'c:/_data/_save_npy/'

x = np.load(np_path + 'keras39_catDog_x_train.npy')
y = np.load(np_path + 'keras39_catDog_y_train.npy')
test = np.load(np_path + 'keras39_catDog_test.npy')
#2

model = Sequential()
model.add(Conv2D(16, (2,2) , strides=2, input_shape=(100, 100, 3)))
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(Dropout(0.4))
model.add(Conv2D(32, (3,3), activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(32, (3,3), activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(units=64))
model.add(Dropout(0.5))
model.add(Dense(128))
model.add(Dense(1, activation='sigmoid'))
model.summary()

#3 
x_train, x_test, y_train, y_test = train_test_split(x, y ,random_state=1452, train_size=0.86, stratify=y)

es = EarlyStopping(monitor='val_loss', mode='min', patience=10, verbose=0, restore_best_weights=True)
model.compile(loss='binary_crossentropy', optimizer='adam' , metrics=['accuracy'])

fit_start_time = time.time()

model.fit(x_train, y_train, batch_size=32, verbose=1, epochs=1000, validation_split=0.2, callbacks=[es])

fit_end_time = time.time()

results = model.evaluate(x_test, y_test)

loss = results[0]
acc = results[1]
# 에측


submit = model.predict(test)
submission = []
submission = submit
threshold = 0.5
binary_submission = (submission > threshold).astype(int)
print(binary_submission.shape) #(5000, 1)
print(binary_submission)

binary_submission = binary_submission.reshape(-1)

#
folder_path = 'C:\\_data\\image\\cat_and_dog\\test2'
file_list = os.listdir(folder_path)
file_names = np.array([os.path.splitext(file_name)[0] for file_name in file_list])

# folder_path = '사진\\폴더\\경로'
# file_list = os.listdir(folder_path)
# file_names = np.array([os.path.splitext(file_name)[0] for file_name in file_list])


y_submit = pd.DataFrame({'id' : file_names, 'Target' : binary_submission})

print(y_submit['Target'])
csv_path = 'C:\\_data\\kaggle\\cat_dog\\'


date = datetime.datetime.now().strftime("%m%d_%H%M")    #01171053   

y_submit.to_csv(csv_path + date + "_acc_" + str(round(acc, 4)) + ".csv", index=False)

print('loss : ' , results[0])
print('acc : ' , results[1])
print("걸린 시간 : ", round(fit_end_time - fit_start_time, 3), '초')

file_acc = str(round(results[1], 6))
import datetime
date = datetime.datetime.now().strftime("%m%d_%H%M")

model.save('C:\\_data\\_save\\models\\kaggle\\cat_dog\\'+ date + '_' + file_acc +'_cnn.hdf5')