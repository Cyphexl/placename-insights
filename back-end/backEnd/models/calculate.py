import os
import re
from pandas import read_csv, merge


def statistic(country):
    rootdir = "./dataset"
    lists = os.listdir(rootdir)
    print(country)
    result={}
    for csv in lists:
        data = read_csv("./dataset/" + csv, index_col=0)
        data = data.fillna(axis=1, method="ffill")
        data = data.fillna(value=0)
        if country in data.index:
            result[csv[:-4]]={}
            size=len(data.loc[country])
            print(size)
            for i,v in enumerate(list(data.loc[country])[-10:]):
                if size+i-10<0:
                    year=data.loc[country].index[i]
                else:
                    year=data.loc[country].index[size+i-10]
                result[csv[:-4]][year]=v
    print(result)
    return result

def countries():
    lists = read_csv('./dataset/country.csv', usecols=[3])
    res=[]
    for i in list(lists.values):
        p1 = re.compile('(.*), (.*)')
        p2 = re.compile('(.*)\.(.*)')
        if not (p1.match(i[0]) or p2.match(i[0])):
            res.append(i[0])
    return res

