# -*- coding: GBK -*-
import os
import numpy as np
from pandas import read_csv, merge
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.model_selection import cross_val_predict

rootdir="../dataset"
list = os.listdir(rootdir)
country=read_csv("../country.csv")

for csv in list:
    # print(csv)
    data=read_csv("../dataset/"+csv,index_col=0)
    data=read_csv("../dataset/"+csv,index_col=0,usecols=[0,len(data.columns)-1],names=['name',csv[:-4][:3]])
    data=data.fillna(axis=1,method="ffill")
    data=data.fillna(value=0)
    # print(csv)
    country=merge(country,data,on="name")

# print(country.columns)

x = country[['ene', 'tax', 'for', 'imp', 'agr_x', 'inc', 'chi', 'agr_y',]]
# print(x.head())

y = country[['gdp']]
# print(y.head())

X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=1)
linreg = LinearRegression()
linreg.fit(X_train, y_train)
print(linreg.intercept_)
print(linreg.coef_)

y_pred = linreg.predict(X_test)
print("MSE:", metrics.mean_squared_error(y_test, y_pred))
print("RMSE:", np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

predicted = cross_val_predict(linreg, x, y, cv=10)
fig, ax = plt.subplots()
ax.scatter(y, predicted)
ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()

