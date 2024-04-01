import numpy as np
from sklearn.datasets import load_linnerud
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.multioutput import MultiOutputRegressor

#1 data
x,y = load_linnerud(return_X_y=True)
print(x.shape, y.shape) #(20, 3) (20, 3)
#최종값-> :[2. 110 . 43]], y: [138. 33. 68.]]

#234 model
model = RandomForestRegressor()
model.fit(x,y)
y_pred = model.predict(x)
print(model.__class__.__name__, 'score:', round(mean_absolute_error(y,y_pred),4)) #RandomForestRegressor score: 3.6953
print(model.predict([[2,110,43]])) #[[153.15  34.28  64.  ]]

#234 model
model = Ridge()
model.fit(x,y)
y_pred = model.predict(x)
print(model.__class__.__name__, 'score:', round(mean_absolute_error(y,y_pred),4)) #Ridge score: 7.4569
print(model.predict([[2,110,43]])) #[[187.32842123  37.0873515   55.40215097]]

#234 model
model = LinearRegression()
model.fit(x,y)
y_pred = model.predict(x)
print(model.__class__.__name__, 'score:', round(mean_absolute_error(y,y_pred),4)) #LinearRegression score: 7.4567
print(model.predict([[2,110,43]]))# [[187.33745435  37.08997099  55.40216714]]

#234 model
model = XGBRegressor()
model.fit(x,y)
y_pred = model.predict(x)
print(model.__class__.__name__, 'score:', round(mean_absolute_error(y,y_pred),4)) #XGBRegressor score: 0.0008
print(model.predict([[2,110,43]]))# [[138.0005    33.002136  67.99897 ]]

model = MultiOutputRegressor(LGBMRegressor(verbose=0))
model.fit(x,y)
y_pred = model.predict(x)
print(model.__class__.__name__, 'score:', round(mean_absolute_error(y,y_pred),4)) #MultiOutputRegressor score: 8.91
print(model.predict([[2,110,43]])) #[[178.6  35.4  56.1]]

model = MultiOutputRegressor(CatBoostRegressor(verbose=0))
model.fit(x,y)
y_pred = model.predict(x)
print(model.__class__.__name__, 'score:', round(mean_absolute_error(y,y_pred),4)) #MultiOutputRegressor score: 0.2154
print(model.predict([[2,110,43]])) #[[138.97756017  33.09066774  67.61547996]]

model = MultiOutputRegressor(loss_function = 'MultiRMSE')
model.fit(x,y)
y_pred = model.predict(x)
print(model.__class__.__name__, 'score:', round(mean_absolute_error(y,y_pred),4)) #MultiOutputRegressor score: 0.2154
print(model.predict([[2,110,43]])) #[[138.97756017  33.09066774  67.61547996]]

model = MultiOutputRegressor(loss_function = 'MultiMSE')
model.fit(x,y)
y_pred = model.predict(x)
print(model.__class__.__name__, 'score:', round(mean_absolute_error(y,y_pred),4)) #MultiOutputRegressor score: 0.2154
print(model.predict([[2,110,43]])) #[[138.97756017  33.09066774  67.61547996]]