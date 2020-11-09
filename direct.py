
# %%
import numpy as np
from sklearn.linear_model import LinearRegression

x1 = np.random.randn(100)
x2 = np.random.randn(100)
er = np.random.randn(100)
X = np.vstack([np.ones(100), x1, x2]).T

# y = 2 + 5 * x1 + 7 * x2 + er
w = np.array([2, 5, 7]).T
y = X @ w + er

B = np.linalg.inv (X.T @ X) @ X.T @ y
print (B)

model = LinearRegression()
model.fit(X, y)
print(model.intercept_, model.coef_)

# %%
