'''
import numpy as np
import random
import os
from IPython.display import display

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

seed_everything(42) # Seed 고정

import pandas as pd

train = pd.read_csv('C:\_data\dacon\predict_income\open//train.csv')
test = pd.read_csv('C:\_data\dacon\predict_income\open//test.csv')

display(train.head(3))
display(test.head(3))

train_x = train.drop(columns=['ID', 'Income'])
train_y = train['Income']

test_x = test.drop(columns=['ID'])

from sklearn.preprocessing import LabelEncoder

encoding_target = list(train_x.dtypes[train_x.dtypes == "object"].index)

for i in encoding_target:
    le = LabelEncoder()
    
    # train과 test 데이터셋에서 해당 열의 모든 값을 문자열로 변환
    train_x[i] = train_x[i].astype(str)
    test_x[i] = test_x[i].astype(str)
    
    le.fit(train_x[i])
    train_x[i] = le.transform(train_x[i])
    
    # test 데이터의 새로운 카테고리에 대해 le.classes_ 배열에 추가
    for case in np.unique(test_x[i]):
        if case not in le.classes_: 
            le.classes_ = np.append(le.classes_, case)
    
    test_x[i] = le.transform(test_x[i])
    
# from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor,  AdaBoostRegressor

# # model = RandomForestRegressor() 
model = GradientBoostingRegressor(random_state=31234)
# model = AdaBoostRegressor()


# model = DecisionTreeRegressor() 
model.fit(train_x, train_y) 
preds = model.predict(test_x)

submission = pd.read_csv('C:\_data\dacon\predict_income\open//sample_submission.csv')
submission['Income'] = preds
submission

submission.to_csv('C:\_data\dacon\predict_income\open//submission0321_1show.csv', index=False)
'''

import numpy as np
import pandas as pd
import random
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from IPython.display import display

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

seed_everything()  # Seed fixed

# Assuming your data is now located in a relative path
train = pd.read_csv('C:\_data\dacon\predict_income\open//train.csv')
test = pd.read_csv('C:\_data\dacon\predict_income\open//test.csv')

display(train.head(3))
display(test.head(3))

train_x = train.drop(columns=['ID', 'Income'])
train_y = train['Income']
test_x = test.drop(columns=['ID'])

# Handle missing values
for column in train_x.columns:
    if train_x[column].dtype == 'object':  # Categorical
        train_x[column] = train_x[column].fillna('Unknown')
        test_x[column] = test_x[column].fillna('Unknown')
    else:  # Numeric
        train_x[column] = train_x[column].fillna(train_x[column].median())
        test_x[column] = test_x[column].fillna(train_x[column].median())

# Label encoding for categorical features
encoding_target = list(train_x.dtypes[train_x.dtypes == "object"].index)
for i in encoding_target:
    le = LabelEncoder()
    # Fit on train data
    train_x[i] = train_x[i].astype(str)
    le.fit(train_x[i])
    # Transform both train and test data
    train_x[i] = le.transform(train_x[i])
    test_x[i] = test_x[i].astype(str).map(lambda s: 'Unknown' if s not in le.classes_ else s)
    le.classes_ = np.append(le.classes_, 'Unknown')
    test_x[i] = le.transform(test_x[i])

# Model: Using GradientBoostingRegressor as an example
model = GradientBoostingRegressor(random_state=4567)

# Example of a simple Grid Search for hyperparameter tuning (optional, can be expanded based on needs)
param_grid = {
    'n_estimators': [100, 200,300,400,500,600],
    'learning_rate': [0.01,0.05, 0.1,0.2,0.3],
    'max_depth': [3,4,5,6,7,8],
}
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, n_jobs=-1, scoring='neg_mean_squared_error')
grid_search.fit(train_x, train_y)
best_model = grid_search.best_estimator_

# Predictions
preds = best_model.predict(test_x)

submission = pd.read_csv('C:\_data\dacon\predict_income\open//sample_submission.csv')
submission['Income'] = preds
submission

submission.to_csv('C:\_data\dacon\predict_income\open//submission0328_2.csv', index=False)


# submission = pd.DataFrame({'ID': test['ID'], 'Income': preds})

# # Save submission
# submission_path = './data/submission_combined_version.csv'
# submission.to_csv(submission_path, index=False)
