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

The GeoNames geographical database contains over 10 million geographical names and consists of over 9 million unique features with 2.8 million populated places and 5.5 million alternate names. All features are categorized into one out of nine feature classes and further subcategorized into one out of 645 feature codes.

## Use-case Diagrams
For this part, we designed a use case diagram to show the relationship between our various use cases, such as the relationship between the backend and the user, the relationship between the front end and the user, and some interaction between them.

------

![下载.png](https://i.loli.net/2019/06/11/5cffb2ee1403846264.png)

*fig - the use case diagram*

## Class Diagrams

Immediately after we created the class diagram to represent the relationship between our various classes, we designed six classes, namely: Doa, MachineLearning, Statistics, Browser, Classfication, Regression. Through these six classes to expand our specific jobs.

------

![下载 (1).png](https://i.loli.net/2019/06/11/5cffb782e15e421785.png)

*fig - the class diagram*

## Sequential Diagrams

The Sequential diagram is a diagram that shows the relationship between our specific projects. It describes the complete flow and interaction details of our entire program by describing the operations between the front end, the back end, and the user.

We have designed two Sequential diagrams to represent our two initial ideas for the overall architecture of the program. Finally, we will start our work based on these two Sequential diagrams.

------

![下载 (2).png](https://i.loli.net/2019/06/11/5cffb900bfb5e81168.png)

*fig - the first squence diagram*

![下载 (3).png](https://i.loli.net/2019/06/11/5cffb900c2d9661659.png)

*fig - the second sequence diagram*



## Global Architecture

Considering that the main machine learning task contains only a single input and output, the project is not too complicated at the architectural level. We are suggested to put a part of data visualization results and analysis in the application interface also. This project can be split into the following modules from the perspective of global architecture:

### GUI Module

The frontend uses the Web as the application interface because the Web has become the only *de-facto* cross-platform, universal interface standard in the IT industry. We use HTML to complete the markup documentation, SASS to write styles, JavaScript to implement web requests and interaction logic, and to automate development through Gulp-like front-end modern workflow tools. The source code is compiled into a static web file via Gulp, and served afterward.

### Machine Learning Module

This section contains clustering, classification, and regression, where the output of the cluster is input to the classification. Regression is used only for model comparison in theoretical research and does not have a significant impact on the function of the application. Most of the code for this module is written using Python and related machine learning libraries. It is worth noting that the neural network after machine learning training is stored as a cache on the server, rather than training a new model separately each time the user requests it. We believe this helps maintain performance consistency of our application and saves computing resources.

### API Module

This module is responsible for the API of the program, which works as a bridge communicating the results of machine learning and the input/output of the graphical user interface. We use Flask to implement this backend part of the web development.

### Continuous Integration Module

Deploying continuous integration helps automate the entire software development process, eliminating the need for manual testing, compilation, and deployment for each update. We use Jenkins to complete the CI module and capture real-time code updates uploaded to GitHub via tools including Webhook. All modules are deployed on the same Amazon web server and properly decoupled.




# II. Elaboration step

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

### Deficiency Detection

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

`POST /city/predict`

Get the input city name and do the predict, return the prediction's result including three most probable areas.

`POST /country/statistic`

Get the input country name and stat the last ten years data.

`GET /country/list`

Return all the countries' names.

## GUI Static Implementation



# III-a. Machine Learning

## Clustering

Before machine learning, we need to cluster this city by  so we choose K-means model to do this job. Considering the vast difference between different countries, like location, culture, population, language, GDP and so on, so we use the Gapminder to do the K-means method, and we need to use the **_PCA_** method to do dimension reduction analysis.

```python
data = normalize(np.array(netArr), axis=0)
pca = PCA(n_components=2)
pca.fit(data)
afterData = pca.fit_transform(data)
```

After dimension reduction, we need to decide the value of **_K_** , we use **_sum of the squared errors_** and **_Silhouette analysis_** to decide the choose amount. Here are math theories of these two methods.

**_sum of the squared errors_**

```latex
SSE=\sum_{i=1}^{K}{\sum_{p∈Ci}{|p-mi|^2}}
```

**_Silhouette analysis_**

```latex
s(i)=\frac{b(i)-a(i)}{max\{a(i),b(i)\}}\quad s(x)=\left
\{\begin{aligned}
1-\frac{a(i)}{c(i)},\quad a(i)<b(i) \\
0									,\quad a(i)=b(i) \\
\frac{a(i)}{c(i)}-1,\quad a(i)>b(i) 
\end{aligned}
\right.
```

Here are our result pictures.

![image-20190519215141458.png](https://i.loli.net/2019/06/12/5cfff0c2e71ca19360.png)

![QQ20190521-0.jpeg](https://i.loli.net/2019/06/11/5cffc9614d6c417843.jpeg)

![image-20190521171733913.png](https://i.loli.net/2019/06/11/5cffc961882f531346.png)

```
For n_clusters = 2 The average silhouette_score is : 0.3905449300354942
For n_clusters = 3 The average silhouette_score is : 0.4176882525682744
For n_clusters = 4 The average silhouette_score is : 0.4461314443682128
For n_clusters = 5 The average silhouette_score is : 0.47883341308085464
For n_clusters = 6 The average silhouette_score is : 0.4165650982422084
For n_clusters = 7 The average silhouette_score is : 0.38732557724848593
For n_clusters = 8 The average silhouette_score is : 0.42761880928564716
For n_clusters = 9 The average silhouette_score is : 0.4719416932136283
```

We' ve got one **_SSE_** picture and 9 **_Silhouette_** pictures, after comparing, **_K=5_** and **_K=9_** can be chosen. But 5 be not enough for classify these areas, so we choose **_K=9_** as our value.

```python
nClusters = 9
kmeans = KMeans(n_clusters=nClusters, random_state=0).fit(afterData) clusteredArr = []
for i in range(0, nClusters): 	 
		clusteredArr.append([])
id = 0
for country in netDict:
		clusteredTo = kmeans.predict([afterData[id]])[0] 	
		clusteredArr[clusteredTo].append(country)
# print("%s in Cluster %s" % (country, kmeans.predict([afterData[id]])))
		id += 1 
```

After clustering, to be compatible with _Geonames_ and _Gapminder_, we use the **_country code_** to describe these countries.

```
EastAsia CN HK JP KP KR LA MO TW VN 
S&SEAsia BD BT BN CC ID IN KH LK MM MV MY NP PH SG TH TL 
EnUsAuNz AU CA CX FK IM IO NZ US VG VI
Latinos AG AI AR AW BB BL BO BR BZ CL CO CR CU CW DM DO EC ES GB GD GI GN GQ GT GY HN HT JM MX NI PA PE PR PT PY SR ST SV TT UY VE
Arabics AE AF BH DZ EG EH IL IQ IR JO KG KW KZ LB LY OM PK PS QA SA SY TJ TM UZ YE  
WEurope AD AL AT BE CH DE DK FI FO FR GL GR HR IE IS IT LI LU MC MT NL NO RE RO SE SM VA
EEurope AM AZ BA BG BA BY CY CZ EE GE HU LT LV MD ME MK MN PL RS RU SI SK UA XK 
Oceania AS BM CK FJ FM KI NR PG PW TK TO TV WS 
SSAfrica AO BF BI BJ BW CD CF CG CI CM CV DJ ER ET GA GH GM GW KE KM LR LS MA MG ML MR MU MW MZ NA NE NG RW SC SD SL SN SO SS SZ TD TG TN TZ UG ZA ZM ZW 
```

Above are the nine areas we clustered, and the next step we need to normalize our city name to tensor.

## Classification

## Regression

For this part, we will use the regression model to train our data. Although the regression is not suitable for our data for application development, we still need to carry out regression analysis and get key data such as MSE and RMSE.

### Merge data set

Firstly，we combine other country-related data sets with previous geographic data sets and get a new data sets which have many information.

Then we select ‘GDP’ as the variable that we want to predict, and eight related variables as our input.

![图片 1.png](https://i.loli.net/2019/06/11/5cffc9ba5b22564401.png) 

*fig - country-related data sets*

```python
rootdir="../dataset"
list = os.listdir(rootdir)
country=read_csv("../country.csv")

for csv in list:
    data=read_csv("../dataset/"+csv,index_col=0)
    data=read_csv("../dataset/"+csv,index_col=0,usecols=[0,len(data.columns)-1],names=['name',csv[:-4][:3]])
    data=data.fillna(axis=1,method="ffill")
    data=data.fillna(value=0)
    country=merge(country,data,on="name")
```

*code - the code of merge*



### 2.MSE and RMSE

When we get the train sets and test sets, we consider using six models to make a regression: 

- **“LinearRegression”**
- **“DecesionTree”**
- **“Kneibor”**
- **“AdaBoostRegressor”**
- **“GBRTRegression(GradientBoosting)”**
- **“ExtraTree”**

For each model we give the Contrast Curve and their Score ，and input their MSE and RMSE:

| Method                |                      MSE |          RMSE           |
| :-------------------- | -----------------------: | :---------------------: |
| **DecesionTree**      |  **0.08550686781742495** | **0.29241557382845557** |
| **LinearRegression**  |  **0.25283456529873943** | **0.5028265757681663**  |
| **Kneibor**           |  **0.01721613900881275** | **0.13121028545359067** |
| **AdaBoostRegressor** |   **0.0636497279190832** | **0.25228897700669206** |
| **GBRTRegression**    | **0.013758100596726035** | **0.11729492997025079** |
| **ExtraTree**         |   **0.0876803445908834** | **0.29610867023929477** |

```python
def try_different_method(model, method):
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)

    y_pred = model.predict(X_test)
    print("MSE:", metrics.mean_squared_error(y_test, y_pred))
    print("RMSE:", np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

    plt.figure()
    plt.plot(np.arange(len(y_pred)), y_test, "go-", label="True value")
    plt.plot(np.arange(len(y_pred)), y_pred, "ro-", label="Predict value")
    plt.title(f"method:{method}---score:{score}")
    plt.legend(loc="best")
    plt.show()
```

*fig - the function of using different methods*



### 3. Contrast Diagrams

Judging from the result, we can notice that most of the models achieve high accuracies, among which GBRTRegression reaches the highest accuracy of 0.99. However, the visualization suggests possible problems of overfitting. Which one appears to be the best model is still to be discussed.

![ 2.png](https://i.loli.net/2019/06/12/5d000a841d6d077778.png)

*fig - LinearRegression*

![ 1.png](https://i.loli.net/2019/06/12/5d000a84627a513885.png)

*fig - DecisionTree*

![3.png](https://i.loli.net/2019/06/12/5d000a8460a1185145.png)

*fig - KNeighbor*

![4.png](https://i.loli.net/2019/06/12/5d000a848a3d725217.png)

*fig - AdaBoostRegression*

![5.png](https://i.loli.net/2019/06/12/5d000a848f00832761.png)

*fig - GBRTRegression*

![6.png](https://i.loli.net/2019/06/12/5d000a849129089404.png)

*fig - ExtraTree*


# III-b. Details & Visualization

## Theoretical Details

## Visualization

# III-c. Programming the Application

## Backend

Our project uses a web application to show our project effect. So we choose front and back separation to make sure our work to be quicker. 

**Flask** is a micro web framework written in Python. It is classified as a micro-framework because it does not require particular tools or libraries. It has no database abstraction layer, form validation, or any other components where pre-existing third-party libraries provide common functions. So it's good enough to make our project better. 

<img src="https://i.loli.net/2019/06/12/5cfff72d0ef8789336.png"  height="200" width="200">

In the Python 3.7 environment, we build three API for the application to show the whole project. We can see  the details in the API documents part.

There are three parts in the backend code. First is the datasets directory, which is used to statistic some data. Next is the model directory,  these model help us predict the location of city name inputted and do some statistics about the different countries.  

Use the below command to run the backend code.

```shell
python3 app.py
```

## Frontend

## Integration

# IV. Deployment and Reporting

## Deployment

## Software Testing

## Continuous integration (CI)
