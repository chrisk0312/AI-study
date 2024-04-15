import numpy as np
from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input, Conv2D, Flatten, MaxPooling2D

#data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000,28*28).astype('float32')/255   # 255 검정 0 백색
x_test = x_test.reshape(10000,28*28).astype('float32')/255  # 255 검정 0 백색

#model
def build_model(drop=0.5, optimizer='adam',activation='relu', node1=128, node2=64, node3=32, lr=0.001):
    inputs = Input(shape=(28*28,), name='input')
    x = Dense(node1, activation=activation, name='hidden1')(inputs)
    x = Dropout(drop)(x)
    x = Dense(node2, activation=activation, name='hidden2')(x)
    x = Dropout(drop)(x)
    x = Dense(node3, activation=activation, name='hidden3')(x)
    x = Dropout(drop)(x)
    output = Dense(10, activation='softmax', name='output')(x)
    
    model = Model(inputs=inputs, outputs=output)
    model.compile(optimizer=optimizer, metrics=['acc'], loss='categorical_crossentropy')
    return model

def create_hyperparameter():
    batches = [100, 200, 300, 400, 500]
    optimizers = ['rmsprop', 'adam', 'adadelta']
    dropout = [0.2, 0.3, 0.4, 0.5]
    activation = ['relu', 'elu', 'selu','linear']
    node1 = [128, 64,32, 16]
    node2 = [128,64,32,16]
    node3 = [128,64,32,16]
    return {"batch_size":batches, "optimizer":optimizers, "drop":dropout, "activation":activation,
            "node1":node1, "node2":node2, "node3":node3}
    
hyperparameters = create_hyperparameter()
print(hyperparameters)

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor
keras_model = KerasClassifier(build_fn=build_model, verbose=1)

model = RandomizedSearchCV(keras_model, hyperparameters, cv=2, 
                           n_iter=10, n_jobs=4, verbose=1)
model.fit(x_train, y_train, epochs = 3)

import time
start_time = time.time()
model.fit(x_train, y_train,epochs=3)
end_time = time.time()
print("time : ",round( end_time - start_time))
print("best_params : ", model.best_params_)
print("best_score : ", model.best_score_)
print('model.score : ', model.score(x_test, y_test))

from sklearn.metrics import accuracy_score
y_pred = model.predict(x_test)
print("acc_score : ", accuracy_score(y_test, y_pred))