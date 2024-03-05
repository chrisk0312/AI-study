import numpy as np  
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error
from sklearn.base import clone
#1 데이터
x,y = load_breast_cancer(return_X_y=True)
x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.8, random_state=777)

scaler = MinMaxScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

parameters = {'n_estimator':1000, 
              'learning_rate':0.1, 
              'max_depth':3, #트리의 최대 깊이
              'gamma':0,
              'min_child_weight':0,
              'subsample':0.4, #전체 데이터의 몇 %를 쓸 것인가 
              'colsample_bytree':0.8, 
              'colsample_bylevel':0.7,
              'reg_alpha':0, #L1 규제, 가중치의 절대값에 대한 페널티,
              'reg_lambda':1, #L2 규제, 가중치의 제곱에 대한 페널티
              'random_state':3377, 
              'verbose':0,
            }
 
#2 모델
model = XGBClassifier()
model.set_params(**parameters)

'''
#3 훈련
model.fit(x_train, y_train, verbose=1, eval_metric=['logloss'], eval_set=[(x_train, y_train),(x_test, y_test)])

#4 평가
results = model.score(x_test,y_test)
print('model.score :', results)

y_pred = model.predict(x_test)
acc = accuracy_score(y_test, y_pred)
print('acc :', acc)

print(model.feature_importances_)

 #for문을 통해 피처가 약한 것부터 하나씩 제거하면서 확인
 #28,27,26....1까지 확인
 # Assume feature_importances contains the feature importances from the trained model
'''
# Placeholder for storing performance metrics
performance_metrics = []

# Start with all features and iteratively remove one feature at a time
num_features = x_train.shape[1]
for i in range(num_features, 0, -1):
    # Train the model
    current_model = clone(model)
    current_model.fit(x_train, y_train, eval_metric='logloss', eval_set=[(x_train, y_train), (x_test, y_test)], verbose=False, early_stopping_rounds=10)
    
    # Evaluate the model
    acc = accuracy_score(y_test, current_model.predict(x_test))
    performance_metrics.append((i, acc))
    
    if i == 1:
        break  # Stop if only one feature is left
    
    # Identify the least important feature and remove it
    feature_importances = current_model.feature_importances_
    least_important_feature = np.argmin(feature_importances)
    x_train = np.delete(x_train, least_important_feature, axis=1)
    x_test = np.delete(x_test, least_important_feature, axis=1)

# Print the performance metrics
for num_features, acc in performance_metrics:
    print(f'Number of features: {num_features}, Accuracy: {acc}')