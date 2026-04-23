"""
Simple Linear Regression example using scikit-learn
"""

import numpy as np
from sklearn.linear_model import LinearRegression

# Sample data
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 6, 8, 10])

# Create and train model
model = LinearRegression()
model.fit(X, y)

# Prediction
prediction = model.predict([[6]])
print("Prediction for x=6:", prediction[0])
