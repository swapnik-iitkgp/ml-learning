"""
Simple Linear Regression Example
--------------------------------
This script demonstrates a basic linear regression workflow using scikit-learn.
"""

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


def main():
    # Generate a simple synthetic dataset
    np.random.seed(42)
    X = 2 * np.random.rand(100, 1)
    y = 4 + 3 * X + np.random.randn(100, 1)

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Create and train the model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)

    print("Linear Regression Model")
    print("----------------------")
    print(f"Intercept: {model.intercept_[0]:.4f}")
    print(f"Coefficient: {model.coef_[0][0]:.4f}")
    print(f"Mean Squared Error: {mse:.4f}")


if __name__ == "__main__":
    main()
