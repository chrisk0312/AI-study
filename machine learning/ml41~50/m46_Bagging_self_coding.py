import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# data
x, y = load_breast_cancer(return_X_y=True)
x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.9,random_state=47)

sclaer = MinMaxScaler().fit(x_train)
x_train = sclaer.transform(x_train)
x_test = sclaer.transform(x_test)

# model 
def bootstrap(data:np.ndarray, data_size, data_num):
    new_data = np.zeros(shape=(data_num,data_size))
    print(new_data.shape)
    import random
    for n in range(data_num):
        for i in range(data_size):
            r = random.randint(0,len(data))
            new_data[n,i] = data[r]
            

def self_random_forest(tree_num,):
    trees = []
    for n in range(tree_num):
        tree = DecisionTreeClassifier()
        trees.append(tree)

print(x_train.shape)
bootstrap(x_train, int(x_train.shape[0]*0.5), 3)
    
    