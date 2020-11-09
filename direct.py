
# %%
import numpy as np
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

n = 100
x1 = np.random.randn(n)
x2 = np.random.randn(n)
er = np.random.randn(n) * 0.2
X = np.vstack([np.ones(100), x1, x2]).T

w = np.array([2, 5, 4]).T
f = X @ w + er
y = 1 / (1 +  np.exp(-f))
print (LogisticFit(X, y))

# B = np.linalg.inv (X.T @ X) @ X.T @ y
# print (B)

# %%
