import numpy as np
from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.model_selection import StratifiedKFold,cross_val_predict
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.utils import all_estimators
from warnings import filterwarnings
filterwarnings('ignore')

#1. data
x,y = load_iris(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, random_state=123, stratify=y)

scaler =MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

n_splits = 5
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123)

allAlgorithms = all_estimators(type_filter='classifier')

print("allAlogrithms :", allAlgorithms)
print("모델의 갯수 :", len(allAlgorithms))

for name, algorithm in allAlgorithms:
    try:
        # model
        model = algorithm()
       
        #3. fit
        cross_val_score(model, x, y, cv=kfold)

        #4. score
        scores = cross_val_score(model, x_train, y_train, cv=kfold)
        print("================================", name, "================================")
        print('ACC:', scores, "\n AVG:", round(np.mean(scores),4))

        y_predict = cross_val_predict(model, x_test, y_test, cv=kfold)
 

        acc = accuracy_score(y_predict,y_test)
        print("acc:", acc)
    except:
        print(name, "은 없음!")
        continue
