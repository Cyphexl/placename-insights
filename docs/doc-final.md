<img src="https://raw.githubusercontent.com/Cyphexl/placename-insights/master/assets/report.svg?sanitize=true">

In the past few decades, the contribution of computers to social productivity mainly lies in its ability to process structural data, such as numerical values, forms, and single colors. People can convert real-life data into structural data and store it in a computer. This liberates a large number of the simplest repetitive workforce.

In recent years, *neural networks* have increased the ability of computers to process non-structural data, or semi-structural data, such as music, images, and text paragraphs; This made abstracting and extracting features of large-scale data possible.

A very important application of machine learning and neural networks is Natural Language Processing. Unlike programming languages, natural language is often more complex, bulky, and difficult to logically describe. The IT industry has introduced abundant research and solutions in this field. When we talk about natural language processing, however, it is usually about a semantic level of processing. In addition to this, there is an area that could be easily overlooked - the abstraction of the *spelling features* of different writing systems.

![1.png](https://i.loli.net/2019/06/11/5cffa7c74fbdb95945.png)

*fig - Placenames in West Europe*

When searching for available datasets related to language and geographic information, we found an open database provided by GeoNames that provides information of the names, locations, countries, and attributes of a large number of geographic features covering the earth. In this experiment, we used data sets provided by GeoNames, combined with `PyTorch`, `Matplotlib` and other data processing and machine learning tools to study the connection between geographic location and city name.



# I. Initialization Step

## Problem Description

## About the Data

## Use-case Diagrams

## Sequential Diagrams

## Global Architecture




# II. Elaboration step

## Detailed Architecture

## Data Scraping and Collection

GeoNames is main data set which we will use to predict the location of the input city name. We do the first check about this data set. 

|       | latitude     | longitude     |
| ----- | ------------ | ------------- |
| count | 1.106199e+07 | 1.106199e+07  |
| mean  | 2.807406e+01 | 1.508189e+01  |
| std   | 2.405836e+01 | 7.962589e+01  |
| min   | 9.000000e+01 | -1.799836e+02 |
| 25%   | 1.600928e+01 | -7.173488e+01 |
| 50%   | 3.288333e+01 | 1.885294e+01  |
| 75%   | 4.434470e+01 | 8.174773e+01  |
| max   | 9.000000e+01 | 1.800000e+02  |

```csv
Total amount:
11061987

Deficiency amount:
country code 13767
asciiname 115
longitude 0
latitude 0
name 0
dtype: int64
```

According to the rough check, only 0.1% of the cities lost their country codes, and almost every city has its `asciiname`. So we can ignore those lost their country codes or `asciiname` city to avoid the effection cased by them.

Besides our main predicting goal, the geography data set is used widely. So we choose some data set about many different kinds of countries in the world to analyze and statistics.

- agriculture_GDP.csv
- agriculture_land.csv
- hiv_adults.csv
- children_per_woman.csv
- energy_production.csv
- income_pre_person.csv
- …

These data set have the common feature that the index of the column are the years, and the row index is the country name. So we may use the latest year data in each data set and join them to the `country_location.csv` to get the new data set. 

| Column         | Description                                                  |
| -------------- | ------------------------------------------------------------ |
| geonameid      | integer id of record in geonames database                    |
| name           | name of geographical point (utf8) varchar(200)               |
| asciiname      | name of geographical point in plain ascii characters, varchar(200) |
| alternatenames | alternatenames, comma separated, ascii names automatically transliterated, convenience attribute from alternatename table, varchar(10000) |
| latitude       | latitude in decimal degrees (wgs84)                          |
| longitude      | longitude in decimal degrees (wgs84)                         |
| country code   | ISO-3166 2-letter country code, 2 characters                 |

*Table - Columns of the geoname dataset*

This dataset's valuable columns are `asciiname`, latitude, longitude and country code. But `asciiname` and country code are all strings; we can't visualize it so that we visualized the coordinates of them.

![9689A4283E8FCA53B4220F018E09BC21.jpg](https://i.loli.net/2019/06/11/5cffbda9d892d87287.jpg)

*Fig - City Distribution*



## Data Cleaning and Transformation

### Deficiency delt

There many deficiencies in this dataset, and the country are only around 200, so we can't ignore them,  after discussion, we decide to use latest and exist data to fill the blank after it and fill `0` to the whole blank line.

```csv
code	latitude	longitude	...	chi	agr_y	gdp
12 	AR	-38.416097	-63.616672	 ...	1.863	7.503740	-0.026220
13 	AT  	47.516231	   14.550072 	 ...  	1.828  	 1.528136 	 -4.135938
14 	   AU	 -25.274398 	 133.775136	  ... 	 1.874 	  2.365473 	 -0.646062
15 	   AZ  	40.143105 	  47.576927	  ...  	1.824  	 6.648044  	 7.054164
16   	 BA	  43.915886	   17.679076	  ...	  1.798  	 7.838391	  -2.742990
17    	BB 	 13.193887 	 -59.543198 	 ...  	1.926 	  3.027063	  -5.497907
18    	BD	  23.684994 	  90.356331	  ... 	 1.772 	 18.728447 	  4.625103 	  …..
```

### Word to Vector

To represent a single letter, we use a “one-hot vector” of size `<1 x n_letters>`. A one-hot vector is filled with 0s except for a 1 at index of the current letter, e.g. `"b" = <0 1 0 0 0 ...>`.

To make a word, we join a bunch of those into a 2D matrix `<line_length x 1 x n_letters>`.

That extra one dimension is because PyTorch assumes everything is in batches - we’re just using a batch size of 1 here.

`BAD` may be converted to tensor like:

```python
0 1 0 0 0 0 ... 0
1 0 0 0 0 0 ... 0
0 0 0 1 0 0 ... 0
```
So we can use *one-hot vector* to convert our city name:

```python
import torch

all_letters = "abcdefghijklmnopqrstuvwxyz'`"
n_letters=len(all_letters)

def letterToIndex(letter):
return all_letters.find(letter)

def letterToTensor(letter):
tensor = torch.zeros(1, n_letters)
tensor[0][letterToIndex(letter)] = 1
return tensor

def lineToTensor(line):
tensor = torch.zeros(len(line), 1, n_letters)
for li, letter in enumerate(line):
tensor[li][0][letterToIndex(letter)] = 1
return tensor
```
Next step, we will build our recurrent neural network to analysis the data classification.



## Data Analysis

 We choose these data set from different aspects so that the analysis may be more diverse.

![image-20190524094536128.png](https://i.loli.net/2019/06/11/5cffbda9505b562984.png)

*Fig - Distribution of the variables*

Before machine learning, we need to cluster these cities by so we choose K-means model to do this job. Considering the vast difference between different countries, like location, culture, population, language, GDP and so on, so we use the Gapminder to do the K-means method, and we need to use the *PCA* method to do dimension reduction analysis.

![image-20190519220213594.png](https://i.loli.net/2019/06/11/5cffbda9d831b99877.png)

## GUI Prototyping & Design

## API Document

## GUI Static Implementation



# III-a. Machine Learning

## Clustering

## Classification

## Regression



# III-b. Details & Visualization

## Theoretical Details

## Visualization



# III-c. Programming the Application

## Backend

## Frontend

## Integration



# IV. Deployment and Reporting

## Deployment

## Software Testing

## Continuous integration (CI)