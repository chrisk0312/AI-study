from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.decomposition import PCA

#data
datasets = load_iris()
x,y = load_iris(return_X_y=True)
print(x.shape, y.shape) #(150, 4) (150,)
x_train , x_test, y_train, y_test = train_test_split(x,y,train_size=0.8,random_state=77,stratify=y)
print(np.min(x_train), np.max(x_train))
print(np.min(x_test), np.max(x_test))

# scaler =MinMaxScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)


# Define a list of models
# model = RandomForestClassifier()
model = make_pipeline(MinMaxScaler(),StandardScaler(),PCA(),RandomForestClassifier(),) #2개의 모델을 연결해주는 파이프라인

model.fit(x_train, y_train)
    

results = model.score(x_test, y_test)
print('score:', results)
y_pred = model.predict(x_test)
print('y_pred:', y_pred)

accuracy_score = accuracy_score(y_test, y_pred)
print('accuracy_score:', accuracy_score)

