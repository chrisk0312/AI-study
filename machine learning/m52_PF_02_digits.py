from xgboost import XGBClassifier, XGBRFRegressor
from sklearn.datasets import load_digits
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import HalvingGridSearchCV, HalvingRandomSearchCV
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import accuracy_score, r2_score
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np


x, y = load_digits(return_X_y=True)
from sklearn.preprocessing import PolynomialFeatures
pf = PolynomialFeatures(degree=2, include_bias=False)
x = pf.fit_transform(x)
x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=333, train_size=0.8,
    stratify=y
)

print(x_train.shape,y_train.shape)
print(np.unique(y_train,return_counts=True))

sclaer = MinMaxScaler().fit(x_train)
x_train = sclaer.transform(x_train)
x_test = sclaer.transform(x_test)

# model 
from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from catboost import CatBoostClassifier
stacking = StackingClassifier([
    ('xgb',XGBClassifier()),
    ('RF',RandomForestClassifier()),
    ('LG',LogisticRegression()),
],final_estimator=CatBoostClassifier())

model = Pipeline([('MM',MinMaxScaler()),('PCA',PCA()),('stacking',stacking)])

model.fit(x_train,y_train)
result = model.score(x_test,y_test)
print("ACC : ",result)

# Score:  0.9555555555555556
# ACC:  0.9555555555555556

# VotingClassifier hard 
# ACC:  0.9555555555555556

# VotingClassifier soft 
# ACC:  0.9638888888888889

# StackingClassifier
# ACC :  0.9666666666666667

# PolynomialFeatures
# ACC :  0.9805555555555555

# PolynomialFeatures + PCA
# ACC :  0.9777777777777777
