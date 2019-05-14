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

allLines = buildNet('cities500.csv')
print(allLines[500])
