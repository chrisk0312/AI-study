import numpy as np
from xgboost import XGBClassifier, XGBRegressor
from sklearn.datasets import load_digits
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# 1
x, y = load_digits(return_X_y=True)

x_train, x_test , y_train, y_test = train_test_split(
    x, y, random_state=777, train_size=0.8,
)

scaler = MinMaxScaler()
# scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

parameters = {
    'n_estimators' : 100,
    'learning_rate' : 0.1,
    'max_depth' : 6,
    'min_child_weight' : 10
}


model = XGBClassifier()

model.set_params(
    early_stopping_rounds=30,
    **parameters)

model.fit(x_train, y_train,
          eval_set=[(x_train, y_train),(x_test, y_test)],
          verbose=10,
          )

results = model.score(x_test,y_test)
print('model.score :', results)

################################################################################
# import pickle
# path = 'C:/_data/_save/_pickle_test/'
# pickle.dump(model, open(path+'m39_pickle1_save.dat', 'wb'))


# import joblib
# path = 'C:/_data/_save/_joblib_test/'
# joblib.dump(model, path+'m40_joblib1_save.dat')

path = 'C:/_data/_save'
model = XGBClassifier()
model.load_model(path+'m41_xgb1_save_model.dat')

results = model.score(x_test,y_test)
print('model.score :', results) 

