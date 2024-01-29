from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
import pandas as pd

#1. 데이터
x= np.array(range(1,17))
y= np.array(range(1,17))

x_train, x_test, y_train, y_test = train_test_split(
    x,y, train_size=0.625, shuffle= False, random_state=66
)

print(x_train, y_train)
print(x_test, y_test)


#2. 모델구성
model= Sequential()
model.add(Dense(10, input_dim=1))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(1))

#3. 컴파일,훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=16,
        #  validation_data=(x_val,y_val)
#         Here, x_val and y_val are explicitly provided as the validation data.
# You prepare a separate set of data for validation, and you pass it to the validation_data parameter during the training process.
# This is useful when you have a specific set of data reserved for validation.
         validation_split= 0.2,
#         Here, validation_split is a fractional number (e.g., 0.2) that represents the proportion of the training data to be used for validation.
# The validation data is randomly selected from the training data based on the specified fraction.
# This is useful when you don't have a separate validation set, and you want to split a portion of your training data for validation on the fly.
        verbose = 1
        )

#4. 평가, 예측
loss = model.evaluate(x_test,y_test)
results = model.predict([30])
print("로스:", loss) 
print("[30]의 예측값:", results) 


#로스: 0.46145689487457275
# [30]의 예측값: [[27.871422]]