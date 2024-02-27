parameters = [
    {"n_estimators":[100,200], "max_depth":[6,10,12],
     "min_samples_leaf":[3,10]},#12
    {"max_depth":[6,8,10,12], "min_samples_leaf":[3,5,7,10]},#16
    {"min_samples_leaf":[3,5,7,10], "min_samples_split":[2,3,5,10]},#16
    {"min_samples_split":[2,3,5,10]},#4
    {"n_jobs":[-1,2,4], "min_samples_split":[2,3,5,10]},#12
]

# #This code defines a list of dictionaries, where each dictionary represents a grid of hyperparameters to explore when tuning a machine learning model,
# likely a RandomForestClassifier or another tree-based model, given the specific parameters included. 

# n_estimators: The number of trees in the forest.
# max_depth: The maximum depth of the tree.
# min_samples_leaf: The minimum number of samples required to be at a leaf node.
# min_samples_split: The minimum number of samples required to split an internal node.
# n_jobs: The number of jobs to run in parallel for both fit and predict. If set to -1, then the number of jobs is set to the number of cores.

# The values in the lists are the specific values that the GridSearchCV will explore for each hyperparameter. 
# For example, n_estimators will be explored at 100 and 200, max_depth will be explored at 6, 10, and 12, and so on.

# The GridSearchCV will try all possible combinations of these hyperparameters and choose the combination 
# that gives the best performance according to a specified scoring metric. 
# For example, if you're using accuracy as your scoring metric, GridSearchCV will choose the hyperparameters that give the highest accuracy.

parameters = [
    {"n_estimators":[100,200], "max_depth":[6,10,12],
     "min_samples_leaf":[3,10]},#12
    {"max_depth":[6,8,10,12], "min_samples_leaf":[3,5,7,10]},#16
    {"min_samples_leaf":[3,5,7,10], "min_samples_split":[2,3,5,10]},#16
    {"min_samples_split":[2,3,5,10]},#4
    {"n_jobs":[-1,2,4], "min_samples_split":[2,3,5,10]},#12
]