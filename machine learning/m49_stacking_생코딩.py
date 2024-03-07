import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
import warnings
from catboost import CatBoostClassifier
warnings.filterwarnings('ignore')

# data
x, y = load_breast_cancer(return_X_y=True)
x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.9,random_state=777,stratify=y)

# sclaer = MinMaxScaler()
# x_train = sclaer.transform(x_train)
# x_test = sclaer.transform(x_test)

# model 
xgb = XGBClassifier()
rf = RandomForestClassifier()
lr = LogisticRegression()

model = StackingClassifier(estimators=[('XGB',xgb),('RF',rf),('LR',lr)],
                           final_estimator=CatBoostClassifier(verbose=0), n_jobs=-1, cv=5)

model.fit (x_train,y_train)

y_pred = model.predict(x_test)
print ("model.score:",model.score(x_test,y_test))
print ("스태킹acc:",accuracy_score(y_test,y_pred))