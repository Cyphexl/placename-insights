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

GeoNames is main dataset that we will use to predict the location of the input city name. We did some general stastistics on the dataset:

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

According to the rough check, only 0.1% of the cities lost their country codes, and almost every city has its `asciiname`. So we can ignore those lost their country codes or `asciiname` city to avoid the negative effect cased by them.

Besides our main predicting goal, the geography dataset is also chosen for the previous clustering process. So we chose some dataset containing information about different countries in the world to analyze and do statistics.

- agriculture_GDP.csv
- agriculture_land.csv
- hiv_adults.csv
- children_per_woman.csv
- energy_production.csv
- income_pre_person.csv
- …

*List of available Gapminder datasets*

These datasets have the common feature that the indices of the columns are years, and the primary keys are the country names. So we may use the data of the latest year in each dataset and join them to the `country_location.csv` to generate a new data set. 

| Column         | Description                                                  |
| -------------- | ------------------------------------------------------------ |
| geonameid      | integer id of record in geonames database                    |
| name           | name of geographical point (utf8) varchar(200)               |
| asciiname      | name of geographical point in plain ascii characters, varchar(200) |
| alternatenames | alternatenames, comma separated, ascii names automatically transliterated, convenience attribute from alternatename table, varchar(10000) |
| latitude       | latitude in decimal degrees (wgs84)                          |
| longitude      | longitude in decimal degrees (wgs84)                         |
| country code   | ISO-3166 2-letter country code, 2 characters                 |

*Table - Columns of the generated dataset*

The useful columns in this dataset are `asciiname`, `latitude`, `longitude` and `country code`. But `asciiname` and country code are all strings; we can't visualize it. Instead, we visualized the coordinates of them.

![9689A4283E8FCA53B4220F018E09BC21.jpg](https://i.loli.net/2019/06/11/5cffbda9d892d87287.jpg)

*Fig - City Distribution*



## Data Cleaning and Transformation

### Deficiency delt

There many deficiencies in this dataset, and the number of the countries is only around 200, so we can't ignore them. After discussion, we decided to use the latest existing data to fill the blank and pad with `0` to the whole line.

```csv
code latitude longitude ... chi agr_y gdp
12 AR -38.416097 -63.616672 ... 1.863 7.503740 -0.026220
13 AT 47.516231 14.550072 ... 1.828 1.528136 -4.135938
14 AU -25.274398 133.775136 ... 1.874 2.365473 -0.646062
15 AZ 40.143105 47.576927 ... 1.824 6.648044 7.054164
16 BA 43.915886 17.679076 ... 1.798 7.838391 -2.742990
17 BB 13.193887 -59.543198 ... 1.926 3.027063 -5.497907
18 BD 23.684994 90.356331 ... 1.772 18.728447 4.625103
```

### Word to Vector

To represent a single letter, we use a “one-hot vector” of size `<1 x n_letters>`. A one-hot vector is filled with 0s except for a 1 at index of the current letter, e.g. `"b" = <0 1 0 0 0 ...>`.

To construct a word, we join a bunch of those into a 2D matrix `<line_length x 1 x n_letters>`.

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

We chose these datasets from different aspects so that the analysis would be more diverse.

![image-20190524094536128.png](https://i.loli.net/2019/06/11/5cffbda9505b562984.png)

*Fig - Distribution of the variables*

We do the correlation matrix to show the most relative 10 variables, and we will come to do clustering with these variables.

![image-20190519220213594.png](https://i.loli.net/2019/06/11/5cffbda9d831b99877.png)

*Fig - The colleration matrix for different columns*

## GUI Prototyping & Design

## API Document

## GUI Static Implementation



# III-a. Machine Learning

## Clustering

Before machine learning, we need to cluster these city by  so we choose K-means model to do this job. Considering the huge difference between different country, like location, culture, population, language, GDP and so on, so we use the gapminderto do the K-means method, and we need use the **_PCA_** method to do  dimension reduction analysis.

> ```python
> data = normalize(np.array(netArr),axis=0)
> pca = PCA(n_components=2)
> pca.fit(data)
> afterData = pca.fit_transform(data)
> ```

After dimension reduction, we need to decide the value of **_K_** , we use **_sum of the squared errors_** and **_Silhouette analysis_** to decide the accurate value. Here are math theories of these two method

**_sum of the squared errors_**
$$
SSE=\sum_{i=1}^{K}{\sum_{p∈Ci}{|p-mi|^2}}
$$

**_Silhouette analysis_**
$$
s(i)=\frac{b(i)-a(i)}{max\{a(i),b(i)\}}\quad s(x)=\left\{\begin{aligned}
1-\frac{a(i)}{c(i)},\quad a(i)<b(i) \\
0									,\quad a(i)=b(i) \\
\frac{a(i)}{c(i)}-1,\quad a(i)>b(i) 
\end{aligned}
\right.
$$
Here are our result pictures.

![image-20190519215141458.png](https://i.loli.net/2019/06/12/5cfff0c2e71ca19360.png)

![QQ20190521-0.jpeg](https://i.loli.net/2019/06/11/5cffc9614d6c417843.jpeg)

![image-20190521171733913.png](https://i.loli.net/2019/06/11/5cffc961882f531346.png)

> For n_clusters = 2 The average silhouette_score is : 0.3905449300354942
>
> For n_clusters = 3 The average silhouette_score is : 0.4176882525682744
>
> For n_clusters = 4 The average silhouette_score is : 0.4461314443682128
>
> For n_clusters = 5 The average silhouette_score is : 0.47883341308085464
>
> For n_clusters = 6 The average silhouette_score is : 0.4165650982422084
>
> For n_clusters = 7 The average silhouette_score is : 0.38732557724848593
>
> For n_clusters = 8 The average silhouette_score is : 0.42761880928564716
>
> For n_clusters = 9 The average silhouette_score is : 0.4719416932136283

We' ve got one **_SSE_** picture and 9 **_Silhouette_** pictures, after comparing, **_K=5_** and **_K=9_** can be choosn. But 5 may be not enough for classify these area, so we choose **_K=9_** as our value.

> ```python
> nClusters = 9
> kmeans = KMeans(n_clusters=nClusters, random_state=0).fit(afterData) clusteredArr = []
> for i in range(0, nClusters): 	 
> 		clusteredArr.append([])
> id = 0
> for country in netDict:
> 		clusteredTo = kmeans.predict([afterData[id]])[0] 	
> 		clusteredArr[clusteredTo].append(country)
> # print("%s in Cluster %s" % (country, kmeans.predict([afterData[id]])))
> 		id += 1 
> ```

&emsp;After clustering , in order to be compatible with _Geonames_ and _Gapminder_, we use the **_country code_** to describe these country.

> <font color='grass'>EastAsia</font> CN HK JP KP KR LA MO TW VN 
>
> <font color='grass'> S&SEAsia</font> BD BT BN CC ID IN KH LK MM MV MY NP PH SG TH TL 
>
> <font color='grass'>EnUsAuNz</font> AU CA CX FK IM IO NZ US VG VI
>
> <font color='grass'>Latinos</font> AG AI AR AW BB BL BO BR BZ CL CO CR CU CW DM DO EC ES GB GD GI GN GQ GT GY HN HT JM MX NI PA PE PR PT PY SR ST SV TT UY VE
>
> <font color='grass'>Arabics</font> AE AF BH DZ EG EH IL IQ IR JO KG KW KZ LB LY OM PK PS QA SA SY TJ TM UZ YE  
>
> <font color='grass'>WEurope</font> AD AL AT BE CH DE DK FI FO FR GL GR HR IE IS IT LI LU MC MT NL NO RE RO SE SM VA
>
> <font color='grass'>EEurope</font> AM AZ BA BG BA BY CY CZ EE GE HU LT LV MD ME MK MN PL RS RU SI SK UA XK 
>
> <font color='grass'>Oceania</font> AS BM CK FJ FM KI NR PG PW TK TO TV WS 
>
> <font color='grass'>SSAfrica</font> AO BF BI BJ BW CD CF CG CI CM CV DJ ER ET GA GH GM GW KE KM LR LS MA MG ML MR MU MW MZ NA NE NG RW SC SD SL SN SO SS SZ TD TG TN TZ UG ZA ZM ZW  

&emsp;Here are nine area which we clustered , so next step we need to normalize our city name to tensor.

## Classification

## Regression

# III-b. Details & Visualization

## Theoretical Details

## Visualization

# III-c. Programming the Application

## Backend

Our project uses web application to show our project effect. So we choose front and back separation to make sure our work to be more quick. 

**Flask** is a micro web framework written in Python. It is classified as a microframeworkbecause it does not require particular tools or libraries. It has no database abstraction layer, form validation, or any other components where pre-existing third-party libraries provide common functions. So its good enough to make our project better. 



## Frontend

## Integration

# IV. Deployment and Reporting

## Deployment

## Software Testing

## Continuous integration (CI)