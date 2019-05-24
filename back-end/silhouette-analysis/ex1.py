'''
Problems
- Deprecation of PCA
- Normalization makes IndiaChina in 1 cluster
'''

import csv
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
import numpy as np

def buildNet(filename):
    file0 = open(filename, 'r', encoding='UTF-8')
    reader0 = csv.reader(file0, delimiter=';')

    netDict = {}
    for row in reader0:
        for i in range(3, 8):
            row[i] = float(row[i])
        if (row[2] == '1951'):
            netDict[row[0]] = {}
            netDict[row[0]]['before'] = {}
            netDict[row[0]]['before']['income'] = row[3]
            netDict[row[0]]['before']['health'] = row[4]
            netDict[row[0]]['before']['population'] = row[5]
            netDict[row[0]]['before']['lat'] = row[6]
            netDict[row[0]]['before']['lon'] = row[7]
        if (row[2] == '2008'):
            netDict[row[0]]['after'] = {}
            netDict[row[0]]['after']['income'] = row[3]
            netDict[row[0]]['after']['health'] = row[4]
            netDict[row[0]]['after']['population'] = row[5]
            netDict[row[0]]['after']['lat'] = row[6]
            netDict[row[0]]['after']['lon'] = row[7]
            netDict[row[0]]['increaseRate'] = {}
            for attr in netDict[row[0]]['after']:
               inc = netDict[row[0]]['increaseRate']
               inc[attr] = (float(netDict[row[0]]['after'][attr]) / float(netDict[row[0]]['before'][attr]) - 1) * 100

            
    file0.close()
    return netDict

netDict = buildNet('wealth1951.txt')
netArr = []
for country in netDict:
    dataset = netDict[country]
    increaseRate = 0
    toAppend = []
    toAppend.append(dataset['after']['lat'])
    for attr in dataset['before']:
        toAppend.append(dataset['before'][attr])
        toAppend.append(dataset['after'][attr])
        toAppend.append(dataset['increaseRate'][attr])
    netArr.append(toAppend)

data = normalize(np.array(netArr), axis=0)
pca = PCA(n_components=5)
pca.fit(data)
afterData = pca.fit_transform(data)
# print(pca.explained_variance_ratio_)
# print(afterData)

nClusters = 4
kmeans = KMeans(n_clusters=nClusters, random_state=0).fit(afterData)

clusteredArr = []
for i in range(0, nClusters):
    clusteredArr.append([])

id = 0
for country in netDict:
    clusteredTo = kmeans.predict([afterData[id]])[0]
    clusteredArr[clusteredTo].append(country)
    # print("%s in Cluster %s" % (country, kmeans.predict([afterData[id]])))
    id += 1

print()
print("Clustered Result:")
print()
for cluster in clusteredArr:
    print(cluster)
    print()


