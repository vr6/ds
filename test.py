#%%
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

#%%
x1 = np.random.randn(100)
x2 = np.random.randn(100)
er = np.random.randn(100) * 0.2

y = 2 + 5 * x1 + 7 * x2 + er
X = np.vstack([x1, x2]).T

print(X.shape, x1.shape, x2.shape, er.shape, y.shape)
ax = plt.figure().add_subplot(111, projection="3d")
ax.scatter(x1, x2, y)
# %%

model = LinearRegression()
model.fit(X, y)
print(model.intercept_, model.coef_)

# %%
import torch

# f(x) = b0 + [(b1 * X)]
# model = f
def f(X, b0, b1):
    return b0 + X @ b1
    # 1st layer nn


def loss(X, y, b0, b1):
    error = y - f(X, b0, b1)
    # loss = error @ error.T
    loss = error.abs().sum()  # mae
    return loss


#%%
import torch

# f(x) = b0 + [(b1 * X)]
# model = f
def f(X, b0, b1):
    return b0 + X @ b1
    # 1st layer nn


def loss(X, y, b0, b1):
    error = y - f(X, b0, b1)
    # loss = error @ error.T
    loss = error.abs().sum()  # mae
    return loss


#%%
step_size = 0.001

b0.requires_grad_()
b1.requires_grad_()

# test-train split, keep computing loss on test:
# stop when loss increases on test
for i in range(500):  # epochs, training iterations
    loss_value = loss(X, y, b0, b1)

    loss_value.backward()

    with torch.no_grad():
        b0 -= step_size * b0.grad
        b1 -= step_size * b1.grad
        b0.grad.zero_()
        b1.grad.zero_()

print(b0, b1)
#%%

# Example autograd

x = torch.Tensor([5])
x.requires_grad_()

y = (x ** 2) + (x ** 3)
y.backward()  # differentiate y

print(y, x.grad)x = torch.Tensor([5])
x.requires_grad_()

y = (x ** 2) + (x ** 3)

y.backward() # differentiate y

y, x.grad # gradient at x of ^ derivative at (x=5)  # gradient at x of ^ derivative at (x=5)
