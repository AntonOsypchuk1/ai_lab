import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

m = 100
X = 6 * np.random.rand(m, 1) - 5
y = 0.7 * X ** 2 + X + 3 + np.random.randn(m, 1)

linear_reg = LinearRegression()
linear_reg.fit(X, y)

X_plot = np.linspace(-4, 6, 100).reshape(-1, 1)
y_linear_plot = linear_reg.predict(X_plot)
plt.scatter(X, y, label="Data", alpha=0.6)
plt.plot(X_plot, y_linear_plot, label="Linear Prediction", color="red", linewidth=2)
plt.title("Linear Regression")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.show()

poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly_features.fit_transform(X)

poly_reg = LinearRegression()
poly_reg.fit(X_poly, y)

print("First data point X[0]:", X[0])
print("Transformed features X_poly[0]:", X_poly[0])
print("Polynomial Regression coefficients:", poly_reg.coef_)
print("Polynomial Regression intercept:", poly_reg.intercept_)

y_poly_plot = poly_reg.predict(poly_features.transform(X_plot))
plt.scatter(X, y, label="Data", alpha=0.6)
plt.plot(X_plot, y_poly_plot, label="Polynomial Prediction", color="green", linewidth=2)
plt.title("Polynomial Regression (Degree 2)")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.show()
