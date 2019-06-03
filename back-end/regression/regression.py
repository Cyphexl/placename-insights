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
from sklearn import tree
from sklearn import svm
from sklearn import neighbors
from sklearn import ensemble
from sklearn.tree import ExtraTreeRegressor


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

def f(x1, x2):
    y = 0.5 * np.sin(x1) + 0.5 * np.cos(x2) + 3 + 0.1 * x1
    return y

def load_data():
    x1_train = np.linspace(0,50,500)
    x2_train = np.linspace(-10,10,500)
    data_train = np.array([[x1,x2,f(x1,x2) + (np.random.random(1)-0.5)] for x1,x2 in zip(x1_train, x2_train)])
    x1_test = np.linspace(0,50,100)+ 0.5 * np.random.random(100)
    x2_test = np.linspace(-10,10,100) + 0.02 * np.random.random(100)
    data_test = np.array([[x1,x2,f(x1,x2)] for x1,x2 in zip(x1_test, x2_test)])
    return data_train, data_test

train, test = load_data()
X_train, y_train = train[:,:2], train[:,2] #数据前两列是x1,x2 第三列是y,这里的y有随机噪声
X_test ,y_test = test[:,:2], test[:,2] # 同上,不过这里的y没有噪声
# print(X_test)

# X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=1)


def try_different_method(model, method):
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)

    # print(model.intercept_)
    # print(model.coef_)

    y_pred = model.predict(X_test)
    print("MSE:", metrics.mean_squared_error(y_test, y_pred))
    print("RMSE:", np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

    # predicted = cross_val_predict(model, x, y, cv=10)
    # fig, ax = plt.subplots()
    # ax.scatter(y, predicted)
    # ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
    # ax.set_xlabel('Measured')
    # ax.set_ylabel('Predicted')
    # plt.title(f"method:{method}---score:{score}")
    # plt.show()

    plt.figure()
    plt.plot(np.arange(len(y_pred)), y_test, "go-", label="True value")
    plt.plot(np.arange(len(y_pred)), y_pred, "ro-", label="Predict value")
    plt.title(f"method:{method}---score:{score}")
    plt.legend(loc="best")
    plt.show()


model_decision_tree_regression = tree.DecisionTreeRegressor()
model_linear_regression = LinearRegression()
model_svm = svm.SVR()
model_k_neighbor = neighbors.KNeighborsRegressor()
model_random_forest_regressor = ensemble.RandomForestRegressor(n_estimators=20)  # 使用20个决策树
model_adaboost_regressor = ensemble.AdaBoostRegressor(n_estimators=50)  # 这里使用50个决策树
model_gradient_boosting_regressor = ensemble.GradientBoostingRegressor(n_estimators=100)  # 这里使用100个决策树
model_extra_tree_regressor = ExtraTreeRegressor()

try_different_method(model_decision_tree_regression, "DecesionTree")
try_different_method(model_linear_regression, "LinearRegression")
# try_different_method(model_svm, "SVMRegression")
try_different_method(model_k_neighbor, "KNeibor")
# try_different_method(model_random_forest_regressor, "RandomForest")
try_different_method(model_gradient_boosting_regressor, "GBRTRegression")
try_different_method(model_extra_tree_regressor, "ExtraTree")

