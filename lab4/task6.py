import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

m = 100
X = 6 * np.random.rand(m, 1) - 5
y = 0.7 * X**2 + X + 3 + np.random.randn(m, 1)

def plot_learning_curves(model, X, y):
    """
    Plots learning curves for a given model, showing training and validation errors
    as the training set size increases.
    
    Parameters:
        model: The machine learning model to evaluate.
        X: Features.
        y: Target values.
    """
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    train_errors, val_errors = [], []
    
    for m in range(1, len(X_train)):
        model.fit(X_train[:m], y_train[:m])
        y_train_predict = model.predict(X_train[:m])
        y_val_predict = model.predict(X_val)
        
        train_errors.append(mean_squared_error(y_train[:m], y_train_predict))
        val_errors.append(mean_squared_error(y_val, y_val_predict))
    
    plt.plot(np.sqrt(train_errors), "r-+", linewidth=2, label="Training error")
    plt.plot(np.sqrt(val_errors), "b-", linewidth=3, label="Validation error")
    plt.title("Learning Curves")
    plt.xlabel("Training Set Size")
    plt.ylabel("RMSE")
    plt.legend()
    plt.show()

linear_reg = LinearRegression()
print("Learning curves for Linear Regression:")
plot_learning_curves(linear_reg, X, y)

polynomial_reg = Pipeline([
    ("poly_features", PolynomialFeatures(degree=10, include_bias=False)),
    ("lin_reg", LinearRegression()),
])
print("Learning curves for Polynomial Regression (Degree 10):")
plot_learning_curves(polynomial_reg, X, y)
