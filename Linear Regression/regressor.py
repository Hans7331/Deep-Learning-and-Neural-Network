

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd



dataset = pd.read_csv('group1_train.txt',sep = ' ')
X_train= dataset.iloc[:, [1]].values
y_train = dataset.iloc[:, 2].values

dataset1 = pd.read_csv('group1_test.txt',sep = ' ')
X_test= dataset1.iloc[:, [1]].values
y_test= dataset1.iloc[:, 2].values

from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X_train)
poly_reg.fit(X_poly, y_train)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y_train)


plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X), color = 'blue')
plt.title('Training set')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, lin_reg_2.predict(poly_reg.fit_transform(X_train)), color = 'blue')
plt.title('Training Set')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()


X_grid = np.arange(min(X_train), max(X_train), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color = 'blue')
plt.title('Test set')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()


X_grid = np.arange(min(X_train), max(X_train), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color = 'blue')
plt.title('Test set')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()