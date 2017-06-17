# -*- coding: utf-8 -*-

#-----------------
# Loading The Data
#-----------------
# Numpy arrays or Scipy sparse matrices or Pandas DataFrame
import numpy as np
X = np.random.random((10,5))
y = np.array(['M','M','F','F','M','F','M','M','F','F'])
X[X < 0.7] = 0

#--------------------------
# Training Data & Test Data
#--------------------------
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

#-----------------------
# Preprocessing The Data
#-----------------------
# Standardization
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler().fit(X_train)
standardized_X_train = scaler.transform(X_train)
standardized_X_test = scaler.transform(X_test)

# Normalization
from sklearn.preprocessing import Normalizer
scaler = Normalizer().fit(X_train)
normalized_X_train = scaler.transform(X_train)
normalized_X_test = scaler.transform(X_test)

# Binarization
from sklearn.preprocessing import Binarizer
binarizer = Binarizer(threshold=0.0).fit(X)
binarized_X = binarizer.transform(X)

# Encoding Categorial Features
from sklearn.preprocessing import LabelEncoder
enc = LabelEncoder()
y = enc.fit_transform(y)

# Imputing Missing Values
from sklearn.preprocessing import Imputer
imp = Imputer(missing_values=0, strategy='mean', axis=0)
imp.fit_transform(X_train)

# Generating Polynomial Features
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(2)
poly.fit_transform(X_train)

#------------------
# Create Your Model
#------------------
# Linear Regression
from sklearn.linear_model import LinearRegression
lr = LinearRegression(normalize=True)

# SVM
from sklearn.svm import SVC
svc = SVC(kernel='linear')

# Naive Bayes
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()

# KNN
from sklearn import neighbors
knn = neighbors.KNeighborsClassifier(n_neighbors=5)

# PCA
from sklearn.decomposition import PCA
pca = PCA(n_components=0.95)

# K Means
from sklearn.cluster import KMeans
k_means = KMeans(n_clusters=3, random_state=0)

#---------------------------
# Model Fitting & Prediction
#---------------------------
# Notice fit() & fit_transform()
# Notice predict() & predict_proba()

#---------------------------------
# Evaluate Your Model's Performace
#---------------------------------
# Classification Metrics - Accuracy Score
knn.score(X_test, y_test)
from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred)

# Classification Metrics - Classification Report
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

# Classification Metrics - Confusion Matrix
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test, y_pred))

# Regression Metrics
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

# Clustering Metrics
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import homogeneity_score
from sklearn.metrics import v_measure_score

# Cross-Validation
from sklearn.cross_validation import cross_val_score
print(cross_val_score(knn, X_train, y_train, cv=4))

#----------------
# Tune Your Model
#----------------
# Grid Search
from sklearn.grid_search import GridSearchCV
params = {"n_neighbors": np.arange(1,3),
          "metric": ["euclidean", "cityblock"]}
grid = GridSearchCV(estimator=knn, param_grid=params)
grid.fit(X_train, y_train)
print(grid.best_score_)
print(grid.best_estimator_.n_neighbors)

# Randomized Parameter Optimization
from sklearn.grid_search import RandomizedSearchCV
params = {"n_neighbors": range(1,5),
          "weights": ["uniform", "distance"]}
rsearch = RandomizedSearchCV(estimator=knn,
                             param_distributions=params,
                             cv=4,
                             n_iter=8,
                             random_state=5)
rsearch.fit(X_train, y_train)
print(rsearch.best_score_)       
