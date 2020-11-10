
# %%
import torch

def logf(X, b):
    return 1 / (1 + torch.exp(-X @ b))

def LogisticFit (X, y):
    xt = torch.Tensor(X)
    yt = torch.Tensor(y)

    # loss = torch.nn.NLLLoss()
    lossfn = torch.nn.BCELoss()
    b = torch.ones(X.shape[1])
    b.requires_grad_()

    # use torch.adam
    step_size = 0.5
    for i in range(2000):
        y_pred = logf(xt, b)
        loss_value = lossfn(y_pred, yt)
        loss_value.backward()
        with torch.no_grad():
            b -= step_size * b.grad
            b.grad.zero_()
    return b

#%%
import numpy as np
import matplotlib.pyplot as plt
n = 100

x1 = np.linspace(-20,10,n)
er = np.random.randn(n) * 2
X = np.vstack([np.ones(n), x1]).T
w = np.array([4, 0.6]).T
f = X @ w
p = 1 / (1 + np.exp(-f))
plt.plot(x1, p, color='blue')

f = f + er
p = 1 / (1 +  np.exp(-f))
plt.scatter(x1, p, color='red')

y = np.where(p > 0.5, 1, 0)
x1 = x1.reshape(n,1)
plt.scatter(x1, y)

from sklearn.linear_model import LogisticRegression
model = LogisticRegression(C=100)
model.fit(x1, y)
c0, c1 = model.intercept_, model.coef_
print(c0, c1)
c = 1 / (1 +  np.exp(-c0 - c1 * x1))
plt.plot(x1, c, color='red')

print (LogisticFit(X, y))

# %%
