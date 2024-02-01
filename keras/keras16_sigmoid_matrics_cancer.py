import numpy as np
from sklearn.datasets import load_breast_cancer
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
import pandas as pd

datasets = load_breast_cancer()
#print(datasets)
print(datasets.DESCR)
print(datasets.feature_names)

x = datasets.data
y = datasets.target
print(x.shape, y.shape) #(569, 30) (569,)


print(np.unique(y)) #[0 1]
print(np.unique(np.array,return_counts=True))# ([212, 357]
print(np.unique(y, return_counts=True))#1 = 357, 0 = 212
print(pd.DataFrame(y).value_counts())#1 = 357, 0 = 212
print(pd.Series(y).value_counts())#1 = 357, 0 = 212
print(pd.value_counts(y))#1 = 357, 0 = 212

#1. 데이터
x = np.array(datasets.data)
y = np.array(datasets.target)

x_train, x_test, y_train, y_test = train_test_split(
    x,y, test_size=0.7, random_state=100
)

print(x_train)
print(y_train)
print(x_test)
print(y_test)

#2. 모델구성
model = Sequential()
model.add(Dense(8, input_dim =30)),#activation="sigmoid"))
model.add(Dense(24))
model.add(Dense(48))
model.add(Dense(96))
model.add(Dense(256))
model.add(Dense(1, activation='sigmoid'))

#3 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam',
              metrics= ['acc','mse','mae']#accuracy =acc
              ) #이진분류에서는 mse 사용안함, binary_crossentropy 사용

from keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val loss',
                   mode = 'min',
                   patience=1,
                   verbose=2,
                   restore_best_weights= True)
hist = model.fit(x_train, y_train, epochs=50, batch_size=32,
                 validation_split=0.9,
                 callbacks =[es]
                 )

#4. 평가,예측
loss = model.evaluate(x_test, y_test) 
print("로스 :", loss)
y_predict = model.predict(x_test)
results = model.predict(x)
np.around(y_predict)
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score
r2= r2_score(y_test, y_predict)
print ("R2 스코어 :",r2)

y_predict = model.predict(x_test)
y_predict = (y_predict > 0.5).astype(int)  # Move this line up

r2= r2_score(y_test, y_predict)
print ("R2 스코어 :",r2)

def acc(y_test, y_predict):
    return(accuracy_score(y_test,y_predict))
rmse =acc(y_test, y_predict)
print("ACC :",acc)

results = model.predict(x)
results = (results > 0.5).astype(int)  # Move this line up

np.around(y_predict)
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score
r2= r2_score(y_test, y_predict)
print ("R2 스코어 :",r2)

def acc(y_test, y_predict):
    return(accuracy_score(y_test,y_predict))
rmse =acc(y_test, y_predict)
print("ACC :",acc)
print("===================================")
#########submission.csv 만들기(count 컬럼에 값만 제출)#############

# Create a new DataFrame for submission
submission_csv = pd.DataFrame()

submission_csv['Outcome'] = np.around(results).flatten()  # Add '.flatten()'
print(submission_csv)
submission_csv.to_csv('c:\_data/samplesubmission.csv', index=False)

import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.figure(figsize=(10,10))
plt.plot(hist.history['loss'],c='red',label = 'loss',marker=".")
plt.plot(hist.history['val_loss'], c='blue',label='val_loss',marker ='.' )
plt.legend(loc='upper right')
plt.title('sigmoid loss')
plt.xlabel('에포')
plt.ylabel('로스')
plt.grid()
plt.show()