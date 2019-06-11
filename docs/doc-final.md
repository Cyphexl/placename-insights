<img src="https://raw.githubusercontent.com/Cyphexl/placename-insights/master/assets/report.svg?sanitize=true">

In the past few decades, the contribution of computers to social productivity mainly lies in its ability to process structural data, such as numerical values, forms, and single colors. People can convert real-life data into structural data and store it in a computer. This liberates a large number of the simplest repetitive workforce.

In recent years, *neural networks* have increased the ability of computers to process non-structural data, or semi-structural data, such as music, images, and text paragraphs; This made abstracting and extracting features of large-scale data possible.

A very important application of machine learning and neural networks is Natural Language Processing. Unlike programming languages, natural language is often more complex, bulky, and difficult to logically describe. The IT industry has introduced abundant research and solutions in this field. When we talk about natural language processing, however, it is usually about a semantic level of processing. In addition to this, there is an area that could be easily overlooked - the abstraction of the *spelling features* of different writing systems.

![1.png](https://i.loli.net/2019/06/11/5cffa7c74fbdb95945.png)

*fig - Placenames in West Europe*

When searching for available datasets related to language and geographic information, we found an open database provided by GeoNames that provides information of the names, locations, countries, and attributes of a large number of geographic features covering the earth. In this experiment, we used data sets provided by GeoNames, combined with `PyTorch`, `Matplotlib` and other data processing and machine learning tools to study the connection between geographic location and city name.



## I. Initialization Step

- Problem Description

- About the Data

- Use-case Diagrams

- Sequential Diagrams

- Global Architecture



## II. Elaboration step

- Detailed Architecture

- Data Scraping and Collection

- Data Cleaning and Transformation

- Data Analysis

- GUI Prototyping & Design

- API Document

- GUI Static Implementation



## III. Construction step

- Machine Learning
  - Clustering
  - Classification
  - Regression

- Theoretical Details

- Visualization

- Programming the Application
  - Backend
  - Frontend
  - Integration



## IV. Deployment and Reporting

- Deployment

- Software Testing

- Continuous integration (CI)



## II. Elaboration step

### Data Scraping and Collection

GeoNames_ is main data set which we will use to predict the location of the input city name. We do the first check about this data set. 

> ```python
>     latitude     longitude
> ```
>
> count **&emsp;** 1.106199e+07  **&emsp;** 1.106199e+07
>
> mean  **&emsp;** 2.807406e+01 **&emsp;** 1.508189e+01
>
> std **&emsp;&emsp;** 2.405836e+01 **&emsp;** 7.962589e+01
>
> min  &emsp;-9.000000e+01&emsp; -1.799836e+02
>
> 25%&emsp;1.600928e+01&emsp;-7.173488e+01
>
> 50%&emsp;3.288333e+01&emsp;1.885294e+01
>
> 75% &emsp;4.434470e+01&emsp;8.174773e+01
>
> max&emsp;9.000000e+01&emsp;1.800000e+02
>
> Total amount:  <font color='red'>11061987</font>
>
> Deficiency amount:
>
> country code&emsp;13767
>
> asciiname&emsp;&emsp;115
>
> longitude&emsp;&emsp;0
>
> latitude&emsp;&emsp;&emsp;0
>
> name&emsp;&emsp;&emsp;&emsp;0
>
> dtype: int64

According to the rough check,  only _0.1%_ city lost their country code, and almost each city has its asciiname. So we can ignore these lost their cc or asciiname city to avoid the effection cased by them.

Besides our main predicting goal, the geography data set is used widely. So we choose some data set about many different kinds of countries in the world to analyze and statistics.

- **agriculture_GDP.csv**
- **agriculture_land.csv**
- **hiv_adults.csv**
- **children_per_woman.csv**
- **energy_production.csv**
- **income_pre_person.csv**
- **…**

These data set have the common feature that the columns index are the years and the row index are the country name. So we may use the lastest year data in each data set and join them to the	**country_location.csv** to get the new data set. 

> - geonameid : integer id of record in geonames database
> - name : name of geographical point (utf8) varchar(200)
> - asciiname : name of geographical point in plain ascii characters, varchar(200)
> - alternatenames : alternatenames, comma separated, ascii names automatically transliterated, convenience attribute from alternatename table, varchar(10000)
> - latitude : latitude in decimal degrees (wgs84)
> - longitude : longitude in decimal degrees (wgs84)
> - country code : ISO-3166 2-letter country code, 2 characters
> - ...

<p align="center"><font  color="#999999" > the columns of the geoname dataset</font></p>

 This data set's valuable columns are asciiname, latitude, longitude and country code. But asciiname and country code are all strings, we can't visualize it, so we just visilize the almost city distribution.

![9689A4283E8FCA53B4220F018E09BC21.jpg](https://i.loli.net/2019/06/11/5cffbda9d892d87287.jpg)

### Data Cleaning and Transformation

##### Deficiency delt

There many deficiency in these dataset ,and the country are only around 200, so we can't ignore them,  after discussion, we decide to use lastest and existed data to fill the blank after it and fill **_0_** to the whole blank line.

> code&emsp;latitude&emsp;longitude&emsp;...&emsp;chi&emsp;agr_y&emsp;gdp
>
> 12 &emsp;AR&emsp;-38.416097&emsp;-63.616672&emsp; ...&emsp;1.863&emsp;7.503740&emsp;-0.026220
>
> 13 &emsp;AT  &emsp;47.516231&emsp;   14.550072 &emsp; ...  &emsp;1.828  &emsp; 1.528136 &emsp; -4.135938
>
> 14 &emsp;   AU&emsp; -25.274398 &emsp; 133.775136&emsp;  ... &emsp; 1.874 &emsp;  2.365473 &emsp; -0.646062
>
> 15 &emsp;   AZ  &emsp;40.143105 &emsp;  47.576927&emsp;  ...  &emsp;1.824  &emsp; 6.648044  &emsp; 7.054164
>
> 16   &emsp; BA&emsp;  43.915886&emsp;   17.679076&emsp;  ...&emsp;  1.798  &emsp; 7.838391&emsp;  -2.742990
>
> 17    &emsp;BB &emsp; 13.193887 &emsp; -59.543198 &emsp; ...  &emsp;1.926 &emsp;  3.027063&emsp;  -5.497907
>
> 18    &emsp;BD&emsp;  23.684994 &emsp;  90.356331&emsp;  ... &emsp; 1.772 &emsp; 18.728447 &emsp;  4.625103 &emsp;  …..

<p align="center"><font  color="#999999" > the brief of the gapminder dataset</font></p>

##### Word to Vector

To represent a single letter, we use a “one-hot vector” of size `<1 x n_letters>`. A one-hot vector is filled with 0s except for a 1 at index of the current letter, e.g. `"b" = <0 1 0 0 0 ...>`.

To make a word we join a bunch of those into a 2D matrix `<line_length x 1 x n_letters>`.

That extra 1 dimension is because PyTorch assumes everything is in batches - we’re just using a batch size of 1 here.

> ​	"bad" may be converted to tensor like:
>
> ​	a b c d e f  … z
>
> ​	0 1 0 0 0 0 ... 0
>
> ​	1 0 0 0 0 0 ... 0
>
> ​	0 0 0 1 0 0 ... 0

&emsp;So we can use **_one-hot vector_** to convert our city name

> ```python
> import torch
> 
> all_letters = "abcdefghijklmnopqrstuvwxyz'`"
> n_letters=len(all_letters)
> 
> def letterToIndex(letter):
>  return all_letters.find(letter)
> 
> def letterToTensor(letter):
>  tensor = torch.zeros(1, n_letters)
>  tensor[0][letterToIndex(letter)] = 1
>  return tensor
> 
> def lineToTensor(line):
>  tensor = torch.zeros(len(line), 1, n_letters)
>  for li, letter in enumerate(line):
>      tensor[li][0][letterToIndex(letter)] = 1
>  return tensor
> ```

&emsp;Next  step, we will build our recurrent neural network to analysis the data classification.

### Data Analysis

 We choose these data set from different aspects, so the analysis may be more diverse.

![image-20190524094536128.png](https://i.loli.net/2019/06/11/5cffbda9505b562984.png)

<p align="center"><font  color="#999999" > distribution of the variables</font></p>

Before machine learning, we need to cluster these city by  so we choose K-means model to do this job. Considering the huge difference between different country, like location, culture, population, language, GDP and so on, so we use the gapminderto do the K-means method, and we need use the **_PCA_** method to do dimension reduction analysis.

![image-20190519220213594.png](https://i.loli.net/2019/06/11/5cffbda9d831b99877.png)

