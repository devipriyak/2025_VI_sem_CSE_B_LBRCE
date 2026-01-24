from sklearn.neighbors import KNeighborsRegressor
import numpy as np

# Training data
X = np.array([[60], [64], [63], [66], [75]])
y = np.array([72, 73, 78, 79, 82])

# Test point
X_test = np.array([[62]])

# Normal KNN Regression
knn_normal = KNeighborsRegressor(
    n_neighbors=5,
    weights='uniform'
)
knn_normal.fit(X, y)
print("Normal KNN Prediction:", knn_normal.predict(X_test))

# Weighted KNN Regression
knn_weighted = KNeighborsRegressor(
    n_neighbors=5,
    weights='distance'
)
knn_weighted.fit(X, y)
print("Weighted KNN Prediction:", knn_weighted.predict(X_test))
