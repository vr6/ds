
# %%
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plot

df = pd.read_csv('Advertising.csv', usecols=[1,2,3,4])

# print (df[0:2])                       # select rows (all columns)
# print (df2[["TV"]])                   # select columns (all rows)
# print (df.iloc[0:2, [0,3]])           # select rows and columns
# print (df[df["TV"] > 250])            # where clause
# print (len(df[df["TV"] > 250]))       # row count
# print (df[df["TV"] > 250].count())    # row count per column

# %%
# X is all data,
# TODO: Split train-set, test-set
# TODO: train on train-set, evaluate on test data, 
# If only using X, then error on train?, error on test?
# ii. X, X^2 -> mse?
# ii. X, X^2 .. X^4 -> mse - overfitting
# Read on overfitting, underfitting. 
#%%
# df.columns
# x = df[['TV', 'Radio', 'Newspaper']].to_numpy()
x = df[['TV']].to_numpy()
y = df.Sales
xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size = 1/3, random_state = 0)
reg = LinearRegression()

#%%  Linear
reg.fit(xTrain, yTrain)
print("intercept-1 =", reg.intercept_)
print("coef-1 =", reg.coef_)

yPred = reg.predict(xTest)
mse = mean_squared_error(yTest, yPred)
print("mse-1 =", mse)

plot.scatter(xTrain, yTrain, color = 'red')
plot.plot(xTrain, reg.predict(xTrain), color = 'blue')
plot.show()
# ==================================================================
#%%  Polynomial X**2
X_poly = np.hstack([xTrain, xTrain * xTrain])
reg.fit(X_poly, yTrain)
print("intercept-2 =", reg.intercept_)
print("coef-2 =", reg.coef_)

X_PolyTest = np.hstack([xTest, xTest * xTest])
yPred = reg.predict(X_PolyTest)
mse = mean_squared_error(yTest, yPred)
print("mse-2 =", mse)

plot.scatter(xTrain, yTrain, color = 'red')
plot.plot(xTrain, reg.predict(X_poly), color = 'blue')
plot.show()

# ==================================================================
#%%  Polynomial X**3
X_poly = np.hstack([xTrain, xTrain * xTrain, xTrain * xTrain * xTrain])
reg.fit(X_poly, yTrain)
print("intercept-3 =", reg.intercept_)
print("coef-3 =", reg.coef_)

X_PolyTest = np.hstack([xTest, xTest * xTest, xTest * xTest * xTest])
yPred = reg.predict(X_PolyTest)
mse = mean_squared_error(yTest, yPred)
print("mse-3 =", mse)

plot.scatter(xTrain, yTrain, color = 'red')
plot.plot(xTrain, reg.predict(X_poly), color = 'blue')
plot.show()

# %%
