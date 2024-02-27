import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
import tensorflow as tf
tf.random.set_seed(777)
np.random.seed(777)
print(tf.__version__) # 2.9.0

# 1. 데이터
x = np.array([1,2])
y = np.array([1,2])

# 2. 모델
model = Sequential()
model.add(Dense(2, input_dim=1))
model.add(Dense(1))

############################################
model.trainable = False#★★★
# model.trainable = True#★★★ #default

############################################
print("=============================================")
print(model.weights) #kernel = weight, bias = bias
print("=============================================")

#compile
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=1000, batch_size=1,verbose=0)

#evaluate
y_pred = model.predict(x)
print(y_pred)