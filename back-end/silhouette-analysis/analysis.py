import os
import numpy as np
from pandas import read_csv, merge
import matplotlib.pyplot as plt
import seaborn as sns


filename="./geonames.csv"
dataset=read_csv(filename,usecols=[1,2,4,5,8],low_memory=False)

print(dataset.describe())

print("Total amount:")
print(len(dataset))
print()
print("Deficiency amount:")
total = dataset.isnull().sum().sort_values(ascending=False)
print(total)


rootdir="../dataset"
list = os.listdir(rootdir)
country=read_csv("../country.csv")

for csv in list:
    # print(csv)
    data=read_csv("../dataset/"+csv,index_col=0)
    data=read_csv("../dataset/"+csv,index_col=0,usecols=[0,len(data.columns)-1],names=['name',csv[:-4][:3]])
    data=data.fillna(axis=1,method="ffill")
    data=data.fillna(value=0)

    country=merge(country,data,left_on='name',right_index=True)


print(country.columns)
corrmat = country.corr()
k = 10 #number of variables for heatmap
cols = corrmat.nlargest(k,'latitude')['latitude'].index
# print cols
cm = np.corrcoef(country[cols].values.T)
sns.set(font_scale=0.9)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f',     annot_kws={'size': 10}, yticklabels=cols.values,    xticklabels=cols.values)
plt.show()

