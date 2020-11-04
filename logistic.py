#%%
import numpy as np
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

n = 100
b0 = 4
b1 = 3

x1 = np.linspace(-5,5,n)
a = 1 / (1 +  np.exp(-b0 - b1 * x1))
plt.plot(x1, a, color='blue')
# blue curve is the actual function

e = np.random.randn(n) * 2
f = b0 + b1 * x1 + e
p = 1 / (1 +  np.exp(-f))
plt.scatter(x1, p, color='red')
# this is fx + random (red dots)

y = np.where(p > 0.5, 1, 0)
x1 = x1.reshape(n,1)
plt.scatter(x1, y)
# these ae the data points on horizontal lines (blue dots)

model = LogisticRegression()
model.fit(x1, y)
c0 = model.intercept_
c1 = model.coef_
print(c0, c1)
c = 1 / (1 +  np.exp(-c0 - c1 * x1))
plt.plot(x1, c, color='red')
