import string
import csv

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
    
    for line in reader:
        line[0]
        counter += 1
        if (counter % 10000 == 0):
            print(counter / 11000000)

    return 0
 

allLines = buildNet('./geonames.csv')
print(allLines[500])

goThroughNet('allCountries.csv')
