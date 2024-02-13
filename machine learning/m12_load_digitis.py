from sklearn.datasets import load_digits
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

datasets = load_digits()
x = datasets.data
y = datasets.target
print(x)
print(y)
print(x.shape, y.shape) #(1797, 64) (1797,)
print(pd.value_counts(y, ascending=True, sort=False))

# 0    178
# 1    182
# 2    177
# 3    183
# 4    181
# 5    182
# 6    181
# 7    179
# 8    174
# 9    180

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=123, stratify=y)

n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=123)

parameters = [
    {"C":[1,10,100,1000], "kernel":["linear"], "degree":[3,4,5]},#12
    {"C":[1,10,100,], "kernel":["rbf"], "gamma":[0.001, 0.0001]},#6
    {"C":[1,10,100,1000], "kernel":["sigmoid"],#24
    "gamma":[0.1,0.001, 0.0001], "degree":[3,4]},
]
model = RandomizedSearchCV(SVC(), parameters, cv=kfold, verbose=1,refit=True, n_jobs=-1,)

# model = GridSearchCV(SVC(), parameters, cv=kfold, verbose=1,refit=True, n_jobs=-1)
model.fit(x_train, y_train)

y_predict = model.predict(x_test)
print("acc:", accuracy_score(y_predict,y_test)) #acc: 0.9916666666666667

y_predict_best = model.best_estimator_.predict(x_test)
print("최적의 매개변수:", model.best_estimator_)
print("최적의 매개변수:", model.best_params_)
print("best_score:", model.best_score_)
print("model.score:", model.score(x_test, y_test))
print(pd.DataFrame(model.cv_results_).T)
