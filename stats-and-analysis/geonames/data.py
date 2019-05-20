import string
import csv
import collections
import re

def buildNet(pathToFile):

    file = open(pathToFile, 'r', encoding='UTF-8')
    reader = csv.reader(file, delimiter='\t')
    
    allLines = []
    tableHeadStr = "geonameid	name	asciiname	alternatenames	lat	lon	feature-class	feature-code	country-code	cc2	a1	a2	a3	a4	population	elevation	dem	timezone	date"
    tableHead = tableHeadStr.split('\t')

    for line in reader:
        currentLine = {}
        for i in range(1, len(tableHead)):
            currentLine[tableHead[i]] = line[i]
        allLines.append(currentLine)

    return allLines

def goThroughNet(pathToFile):

    file = open(pathToFile, 'r', encoding='UTF-8')
    reader = csv.reader(file, delimiter='\t')
    counter = 0

    nCities = {}
    nLetters = {}
    nWords = {}
    nAvLength = {}
    
    for line in reader:
        countryCode = line[8]
        words = re.split('-| |\n', line[2])
        if countryCode in nCities:
            nCities[countryCode] += 1
            nWords[countryCode] += len(words)
            nLetters[countryCode] += len(line[2])
        else:
            nCities[countryCode] = 1
            nWords[countryCode] = len(words)
            nLetters[countryCode] = len(line[2])
        counter += 1
        if (counter % 100000 == 0):
            print(counter / 12000000)

        if (countryCode == 'SJ'):
            print(line[2])

    for country in nCities:
        nAvLength[country] = nLetters[country] / nWords[country]

    nCitiesSorted = sorted((value,key) for (key,value) in nCities.items())
    nAvLengthSorted = sorted((value,key) for (key,value) in nAvLength.items())
    print(nAvLengthSorted)

    return 0
 

allLines = buildNet('cities500.csv')
goThroughNet('allCountries.csv')
