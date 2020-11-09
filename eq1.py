
#%%
import numpy as np

X = np.array([[1,2,3],
                [3,4,5],
                [5,8,7]])

w = np.array([2, 5, 7]).T
y = X @ w
print(np.linalg.inv(X) @ y)

B = np.linalg.inv (X.T @ X) @ X.T @ y
print (B)

# %%
