#basic tools

from ucimlrepo import fetch_ucirepo
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
from itertools import product,permutations,combinations
#R2
from sklearn.metrics import mean_squared_error, r2_score
#degree
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
#import Lasso and LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
#import KNN
from sklearn.neighbors import NearestNeighbors
#k折-交叉驗證
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.linear_model import SGDRegressor

import random
import matplotlib.pyplot as plt
import scipy.optimize as optimization
import statistics
import gc
import csv

df = pd.read_csv('airlinedelaycauses_DelayedFlights.csv')
print(df['DepTime'])
print(df['CRSArrTime'])
X = df['DepTime'][:75]
y = df['CRSArrTime'][:75]
X = np.reshape(X,(-1,1))
y = np.reshape(y,(-1,1))

Ls = Ridge()
param_grid = {'alpha': [1E-4,1E-3,1E-2,1E-1,1E+0,1E+1,1E+2,1E+3,1E+4]} #視情況加入梯度下降
kfold = KFold(n_splits=10, shuffle=True, random_state=40)
grid_search = GridSearchCV(Ls, param_grid, cv=kfold)
grid_search.fit(X,y)



Ls = Ridge(alpha=grid_search.best_params_['alpha'])
model_Ls = Ls.fit(X,y)
predict = model_Ls.predict(X)

rmse_p = np.sqrt(mean_squared_error(y, predict))
r2_p = r2_score(y, predict)

print(f'rmse:{rmse_p}')
print(f'r2:{r2_p}')
plt.scatter(X,y)
plt.plot(X,predict)
plt.show()