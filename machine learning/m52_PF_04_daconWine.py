from xgboost import XGBClassifier, XGBRFRegressor
from sklearn.datasets import fetch_covtype
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import HalvingGridSearchCV, HalvingRandomSearchCV
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import accuracy_score, r2_score
import pandas as pd
import numpy as np

path = "C:\\_data\\DACON\\와인품질분류\\"
train_csv = pd.read_csv(path+"train.csv",index_col=0)
test_csv = pd.read_csv(path+"test.csv",index_col=0)
submit_csv = pd.read_csv(path+"sample_submission.csv")

# print(train_csv.isna().sum(),test_csv.isna().sum()) 결측치 존재안함

# print(train_csv,test_csv,sep='\n') #[5497 rows x 13 columns], [1000 rows x 12 columns]
x = train_csv.drop(['quality'],axis=1)
y = train_csv['quality']

# print(x.shape,y.shape)  #(5497, 12) (5497,)
print(np.unique(y,return_counts=True))
# print(y.shape)          #(5497, 7)

x.loc[x['type'] == 'red', 'type'] = 1
x.loc[x['type'] == 'white', 'type'] = 0
# print(x)
test_csv.loc[test_csv['type'] == 'red', 'type'] = 1 
test_csv.loc[test_csv['type'] == 'white', 'type'] = 0

from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

y = LabelEncoder().fit_transform(y)
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
model = StackingClassifier([
    ('xgb',XGBClassifier()),
    ('RF',RandomForestClassifier()),
    ('LG',LogisticRegression()),
],final_estimator=CatBoostClassifier())

model.fit(x_train,y_train)
result = model.score(x_test,y_test)
print("ACC : ",result)

# Score:  0.5454545454545454
# ACC:  0.5454545454545454

# VotingClassifier hard
# ACC:  0.6763636363636364

# VotingClassifier soft
# ACC:  0.6754545454545454

# StackingClassifier
# ACC :  0.6636363636363637

# PolynomialFeatures
# ACC :  0.6781818181818182