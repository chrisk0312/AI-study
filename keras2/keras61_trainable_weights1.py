import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
import tensorflow as tf
# tf.random.set_seed(777)
np.random.seed(777)
print(tf.__version__) # 2.9.0

# 1. 데이터
x = np.array([1,2,3,4,5])
y = np.array([1,2,3,4,5])

# 2. 모델
model = Sequential()
model.add(Dense(3, input_dim=1))
model.add(Dense(2))
model.add(Dense(1))

model.summary()

print(model.weights) #kernel = weight, bias = bias
print("=============================================")
print(model.trainable_weights) #layer에 있는 weight, bias

print(len(model.weights)) # 6
print(len(model.trainable_weights)) # 6


############################################
model.trainable = False#★★★
############################################

print(len(model.weights)) # 6
print(len(model.trainable_weights)) # 0

print("=============================================")
print(model.weights) #kernel = weight, bias = bias
print("=============================================")
print(model.trainable_weights) #layer에 있는 weight, bias
print("=============================================")

model.summary()
