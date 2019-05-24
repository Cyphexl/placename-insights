from pandas import read_csv
import numpy as np
import torch
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

all_letters="qwertyuiopasdfghjklzxcvbnm[] \;',./"
n_letters=len(all_letters)

np.set_printoptions(suppress=True)

filename="./geonames.csv"
dataset=read_csv(filename,usecols=[1,2,4,5],low_memory=False)

def letterToIndex(letter):
    return all_letters.find(letter)

def letterToTensor(letter):
    tensor = torch.zeros(1, n_letters)
    tensor[0][letterToIndex(letter)] = 1
    return tensor


def lineToTensor(line):
    line=line.lower()
    tensor = torch.zeros(len(line), n_letters)
    for li, letter in enumerate(line):
        tensor[li][letterToIndex(letter)] = 1
    return tensor



for li,line in enumerate(dataset['asciiname']):
    dataset['name'][li]=lineToTensor(line)
    print(dataset['name'][li])

SSE = []  # 存放每次结果的误差平方和
for k in range(1,12):
    estimator = KMeans(n_clusters=k)  # 构造聚类器
    estimator.fit(dataset[['latitude','longitude']])
    SSE.append(estimator.inertia_) # estimator.inertia_获取聚类准则的总和
X = range(1,12)
plt.xlabel('k')
plt.ylabel('SSE')
plt.plot(X,SSE,'o-')
plt.show()



