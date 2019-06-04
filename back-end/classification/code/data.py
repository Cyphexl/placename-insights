import torch
import glob
import unicodedata
import string
import csv

all_letters = "qwertyuiopasdfghjklzxcvb,.:;'- "
n_letters = len(all_letters)


def buildNet(pathToFile):

    file = open(pathToFile, 'r', encoding='UTF-8')
    reader = csv.reader(file, delimiter='\t')

    file0 = open(pathToFile, 'r', encoding='UTF-8')
    reader0 = csv.reader(file0, delimiter='\t')

    file1 = open('../data/clusteredRegions.txt', 'r', encoding='UTF-8')
    reader1 = csv.reader(file1, delimiter=' ')
    
    all_categories_set = set([])
    category_lines = {}
    region_ref = {}
    all_categories = []
    all_lines = []

    for line in reader1:
        region_ref[line[0]] = []
        all_categories.append(line[0])
        for i in range(1, len(line)):
            region_ref[line[0]].append(line[i])

    inverted_ref = {}
    for key, value in region_ref.items():
        for string in value:
            inverted_ref.setdefault(string, []).append(key)

    for line in reader:
        name = line[2].lower()
        country = line[8]
        lat = float(line[4])
        lon = float(line[5])

        if country in inverted_ref:
            region = inverted_ref[country][0]
            category_lines[region] = []
            all_categories_set.add(region)
            all_lines.append([name, region])

    for line in reader0:
        name = line[2].lower()
        lat = float(line[4])
        lon = float(line[5])
        country = line[8]

        if country in inverted_ref:
            region = inverted_ref[country][0]
            category_lines[region].append(name)

    n_categories = len(all_categories)

    file.close()
    file0.close()
    file1.close()

    return category_lines, all_categories, n_categories, all_lines

def findFiles(path): return glob.glob(path)

def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )

category_lines, all_categories, n_categories, all_lines = buildNet('../data/cities15000.txt')

def letterToIndex(letter):
    return all_letters.find(letter)

def lineToTensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        tensor[li][0][letterToIndex(letter)] = 1
    return tensor

