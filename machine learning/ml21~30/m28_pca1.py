# PCA, or Principal Component Analysis, is a dimensionality reduction technique used in machine learning and statistics.
# It's often used when dealing with high-dimensional data, to simplify the dataset while retaining the important information.
# Standardize the data: PCA is affected by the scales of the features, 
# so it's common to standardize the features to have a mean of 0 and a standard deviation of 1.
# Calculate the covariance matrix: The covariance matrix captures the correlation between the different features in the data.
# Compute the eigenvalues and eigenvectors of the covariance matrix: The eigenvectors represent the directions or components in the feature space, 
# while the eigenvalues represent the magnitude or importance of the corresponding eigenvectors.
# Sort the eigenvalues and their corresponding eigenvectors.
# Select the top k eigenvectors: These are the principal components, where k is the number of dimensions of the new feature subspace
# (k<=d, where d is the dimensionality of the original feature space).
# Transform the original dataset: This is done via the selected eigenvectors to obtain a k-dimensional feature subspace.
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import sklearn as sk
from sklearn.ensemble import RandomForestClassifier
print(sk.__version__)#1.1.3

datasets = load_iris()
x = datasets['data']
y = datasets.target
print(x.shape, y.shape) #(150, 4) (150,)

scaler = StandardScaler()
x = scaler.fit_transform(x)

pca = PCA(n_components=1) #n_components= : 몇개로 차원을 축소할 것인지
x = pca.fit_transform(x)
print(x)
print(x.shape) #(150, 4)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=777, shuffle=True, stratify=y)


model = RandomForestClassifier()

model.fit(x_train, y_train)

result = model.score(x_test, y_test)
print(result) #0.9666666666666667

# (150, 4) (150,)
# (150, 4)
# 0.9333333333333333

