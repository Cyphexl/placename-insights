<img src="https://raw.githubusercontent.com/Cyphexl/placename-insights/master/assets/report.svg?sanitize=true">

> The report consists of mainly five sections: the initialization step, the elaboration step, the construction step, the deployment step, and the timeline of work repartition. The construction step is divided into three subsections, demonstrating the work of machine learning, visualization, and application development, respectively. A conclusion is given at the end of the report.



# O. Background & Introduction

In the past few decades, the contribution of computers to social productivity mainly lies in its ability to process structural data, such as numerical values, forms, and single colors. People can convert real-life data into structural data and store it in a computer. This liberates a large number of the simplest repetitive workforce.

In recent years, *neural networks* have increased the ability of computers to process non-structural data, or semi-structural data, such as music, images, and text paragraphs; This made abstracting and extracting features of large-scale data possible.

A very important application of machine learning and neural networks is Natural Language Processing. Unlike programming languages, natural language is often more complex, bulky, and difficult to logically describe. The IT industry has introduced abundant research and solutions in this field. When we talk about natural language processing, however, it is usually about a semantic level of processing. In addition to this, there is an area that could be easily overlooked - the abstraction of the *spelling features* of different writing systems.

![1.png](https://i.loli.net/2019/06/11/5cffa7c74fbdb95945.png)

*fig - Placenames in West Europe*

When searching for available datasets related to language and geographic information, we found an open database provided by GeoNames that provides information of the names, locations, countries, and attributes of a large number of geographic features covering the earth. In this experiment, we used data sets provided by GeoNames, combined with `PyTorch`, `Matplotlib` and other data processing and machine learning tools to study the connection between geographic location and city name.





# I. Initialization Step

## Problem Description

The goal of the project is to abstract the relationship between the spelling of the place name and its geographical location from the GeoNames and GapMinder dataset, and train a recurrent neural network (RNN) on certain goals. The implemented application should demonstrate the insights, statistics, and facts that we discovered, and predict the approximate region of the place name by its spelling. The regions are clustered with the help of GapMinder countries data.

In the later stages of the project, we will explore and summarize the relationship between place name spellings and their coordinates revealed in the training results.

## About the Data

The GeoNames geographical database contains over 10 million geographical names and consists of over 9 million unique features with 2.8 million populated places and 5.5 million alternate names. All features are categorized into one out of nine feature classes and further subcategorized into one out of 645 feature codes.

## Use-case Diagrams
For this part, we designed a use case diagram to show the relationship between our various use cases, such as the relationship between the backend and the user, the relationship between the front end and the user, and some interaction between them.

![下载.png](https://i.loli.net/2019/06/11/5cffb2ee1403846264.png)

*fig - the use case diagram*

## Class Diagrams

Immediately after we created the class diagram to represent the relationship between our various classes, we designed six classes, namely: Doa, MachineLearning, Statistics, Browser, Classfication, Regression. Through these six classes to expand our specific jobs.

![下载 (1).png](https://i.loli.net/2019/06/11/5cffb782e15e421785.png)

*fig - the class diagram*

## Sequential Diagrams

The Sequential diagram is a diagram that shows the relationship between our specific projects. It describes the complete flow and interaction details of our entire program by describing the operations between the front end, the back end, and the user.

We have designed two Sequential diagrams to represent our two initial ideas for the overall architecture of the program. Finally, we will start our work based on these two Sequential diagrams.

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

The graphical user interface, according to the global application structure discussed above, should contain mainly two parts: the machine learning input and output section, and the statistics, insights & visualization section. The interface should be a single-page web application (SPA). Due to the initial design of our project, we do not need to store user data or input, test results. Instead, the interface responds at each anonymous request.

![](https://i.loli.net/2019/06/12/5d00e61b946a758729.png)

*Fig - An early version of the GUI prototype*

The visual design, on the other hand, focuses on typography, layout and visual consistency. Poppins font family is chosen for our identity font, and thus used across the website. We use different font weights to stress the contrast and layering of GUI, and the color magnenta `#df3954` is chosen for theme color. The entire interface is finally implemented on a dark background, which differs from the initial draft above, because the scatter plots look better this way.

![](https://i.loli.net/2019/06/12/5d00e8afee2ae51313.png)

Fig - The implemented version of the application



## API Document

`POST /city/predict`

Get the input city name and do the predict, return the prediction's result including three most probable areas.

`POST /country/statistic`

Get the input country name and stat the last ten years data.

`GET /country/list`

Return all the countries' names.





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

![image-20190612153714930.png](https://i.loli.net/2019/06/12/5d00ac1d37eb752637.png)

*fig - sum of the squared errors*

![image-20190612153849558.png](https://i.loli.net/2019/06/12/5d00ac1df3ec372223.png)

*fig - Silhouette analysis*

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

### Generate the RNN

RNN can be easily derived from the simple feed-forward neural network through PyTorch.

```python
import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)

n_hidden = 128
rnn = RNN(n_letters, n_hidden, n_categories)
```

RNN is a variant of a neural network that still "keep a memory" to the content of the previous input sequence. It has similar input, output, and a hidden layers module. The network is visualized as follows:

![1555167341787](C:\Users\xiaomi\AppData\Roaming\Typora\typora-user-images\1555167341787.png)

### Training the Network

After defining several helper functions, we can begin the main process of training. The purpose of training as a whole is to reduce the cost function by guessing the results and comparing the correct results with feedback and constantly adjusting the parameters of each neuron in the neural network. In this project, `train.py` is the pipeline of the training process, where the `train` function declares a single training process:

```python
learning_rate = 0.003

def train(category_tensor, line_tensor):
    outputt = 0
    loss = 0
    hidden = rnn.initHidden()

    rnn.zero_grad()

    for i in range(line_tensor.size()[0]):
        outputt, hidden = rnn(line_tensor[i], hidden)

    loss = criterion(outputt, category_tensor)
    loss.backward()

    for p in rnn.parameters():
        p.data.add_(-learning_rate, p.grad.data)

    return outputt, loss.item()
```

For each iteration of training, the following process is executed:

- Create input and target tensors
- Create a zeroed initial hidden state
- Read each letter in and Keep hidden state for next letter
- Compare final output to target
- Back-propagate
- Return the output and loss

![1555168661260](C:\Users\xiaomi\AppData\Roaming\Typora\typora-user-images\1555168661260.png)

*fig - The starting of training process. Notice the average loss slowly declining.*



## Regression

For this part, we will use the regression model to train our data. Although the regression is not suitable for our data for application development, we still need to carry out regression analysis and get key data such as MSE and RMSE.

### Merging the data set

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



### MSE and RMSE

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



### Diagrams Comparison

Judging from the result, we can notice that most of the models achieve high accuracies, among which GBRTRegression reaches the highest accuracy of 0.99. However, the visualization suggests possible problems of overfitting. Which one appears to be the best model is still to be discussed.

![](https://i.loli.net/2019/06/12/5d00ea67dfba121891.png)

*fig - Performance on different regression models*






# III-b. Details & Visualization



## Visualization

### Statistics & Facts

#### All Cities

![](https://i.loli.net/2019/05/20/5ce1e8935bd9097588.jpg)

The figure above shows all cities with population more than 500 (total ~180,000 cities) scattered onto a dark background, with each point represents a city. This figure gives a sense of where geographic information is densely aggregated or recorded and where there are not. It seems that western European area holds the most densely distributed populated cities. Other populated area includes the USA, Central America and Southeast China.

```python
    d3.text("cities500.csv").then(function (text) {
        let rows = d3.tsvParseRows(text)
        rows.forEach(function (d) {
            var p = projection([+d[5], +d[4]]);
            if (p) d.x = Math.round(p[0]), d.y = Math.round(p[1]);
        })
        init();
        function init() {
            
            var canvas = d3.select("body").insert("canvas", "input")
                .attr("width", width)
                .attr("height", height)
            var context = canvas.node().getContext("2d");
            context.fillStyle = "#222222";
            context.globalCompositeOperation = 'lighter';
            context.globalAlpha = .8;
            rows.forEach(function (d) {
                context.fillStyle = "#497591";
                context.fillRect(d.x, d.y, 1, 1);
            });
        }
    });
```

#### Cities with Longer Words

![](https://i.loli.net/2019/05/20/5ce1e8940d1c988890.png)

Cities are represented with different colors on this map, and to avoid confusing mixed colors; color blend modes are set to Normal, i.e. no blending. In this figure, blue dot means a city that has short words in its name (e.g. Ho Chi Minh City), and yellow dot indicates long words (e.g. Vladivostok).


#### Cities with Higher Vowel-Consonant Ratios

![](https://i.loli.net/2019/05/20/5ce1e89407c3669605.png)

The idea behind this image is similar to the previous plot, whereas a red dot indicates a city name with more consonants, and a yellow dot indicates more vowels.

#### Cities with More Words

![](https://i.loli.net/2019/05/20/5ce1e893c4b5779918.png)

The idea behind this image is similar to the previous plot, whereas a red dot indicates a city name consists of many words, and a green dot indicates the contrary.

#### Designing a Color Projection Function

To design a proper `words_length -> color` projection function, we firstly investigated the distribution of word lengths. The median word length is around six letters, with a minimum of 1 letter and a maximum of over 20 letters. This is like a normal distribution with *skewed* (or imbalanced) two sides. *Log-normal distribution*, in this case, fits the model.

![450px-PDF-log_normal_distributions.svg.png](https://i.loli.net/2019/05/19/5ce16c3d7c34b27321.png)

Above is its PDF (Probability density function) plotted. To transform the distribution into a color projection function, we need its CDF (Cumulative distribution function) expressions:
```math
CDF(x, \mu, \sigma) = \frac12 + \frac12\operatorname{erf}\Big[\frac{\ln x-\mu}{\sqrt{2}\sigma}\Big]
```
The corresponding part of the code is implemented as below:

```js
    function erf(x) {
        let m = 1.00;
        let s = 1.00;
        let sum = x * 1.0;
        for (let i = 1; i < 50; i++) {
            m *= i;
            s *= -1;
            sum += (s * Math.pow(x, 2.0 * i + 1.0)) / (m * (2.0 * i + 1.0));
        }
        return 2 * sum / Math.sqrt(3.14159265358979);
    }

    function logNormalCDF(x, mu, sigma) {
        let par = (Math.log(x) - mu) / (Math.sqrt(2) * sigma)
        return 0.5 + 0.5 * erf(par)
    }

    const projectColor = (x) => Math.round(logNormalCDF(x/5, 0, 1)*255)
```

Moreover, despite its ugliness, the result color is concatenated from strings:

```js
context.fillStyle = 'rgb(' + projectColor(wordLength) + ',' + projectColor(wordLength) + ',' + (255 - projectColor(wordLength)) + ')'
```



### Finding Pattern In Names

In addition to the regular steps in machine learning and statistics, we tried to find out whether specific "patterns" in place names exist. We limited the area of research to Mainland China for the Chinese language we're both familiar with. We used regular expressions to filter the points in the scatter plot and found out for some specific words or characters, the place names containing them tend to aggregate within a particular area, or show interesting distribution patterns.

```python
            rows.forEach(function (d) {
                let name = d[2]
                let pattern = /DIAN$/i
                if (pattern.test(name)) {
                    context.fillStyle = "#497591";
                    context.fillRect(d.x, d.y, 1, 1);
                }
            });
```

![](<https://i.loli.net/2019/05/20/5ce1e893b966717079.png>)

#### Zhongyuan Markets

"Dian(店)" means "market" in Mandarin Chinese. This is a word that appears only in Zhongyuan Mandarin dialect; therefore, the place names containing "Dian" gather in Zhongyuan region, the central east of Mainland China.

#### Mountains and Plains

"Shan(山)" means "mountain" in Chinese. The place names containing "Shan", as the visualisation suggests, appear more frequently in those mountainous terrains. The western part of China is also considered mountainous, but insufficient in geographic information in general. Still, we can observe that for most populated areas, "Shan"s gather in hills rather than plains. Notice the dense distribution of the hilly regions of Shandong, and the strip of dark areas of the North China Plain surrounding the mountains on the scatter plot.

#### Lakes in the South

"Hu(湖)" stands for "lake" in Mandarin Chinese. Lakes are more densely distributed in Southern China than the north; therefore, in the visualisation, the southern part of China is more densely lit.

#### Tibetan or Minnan Dialect

"Cuo(错)" stands for "lake", too, but in Tibetan instead of Mandarin Chinese. That explains why this time the lakes in Tibet are highlighted. There is a strange, unintentional aggregation of points in the southeast coastline of China though. After investigation, we discovered that there is another Chinese word "Cuo(厝)" that spells identical to "Cuo(错)", which means "House" in the Minnan dialect of the Chinese language. Minnan dialect is widely spoken by inhabitants in the southeast coastline of China, and that thoroughly explains the result.





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

The frontend is written in extended HTML, SASS and ECMAScript6.

### Gulp

![](https://i.loli.net/2019/06/12/5d00eed792f9e88028.png)

Gulp is a toolkit for automating time-consuming tasks in the frontend development workflow. 

### extended HTML

Extended HTML is HTML with specific additional grammars. For instance, to include a inline SVG graphic resource in the HTML document, instead of inserting all SVG path declarations, one may use the following `include` keyword, which is an apparent clearer way:

```html
@include("assets/arrow-left.svg")
```

The compiling is processed in a Gulp pipeline implemented as below:

```js
gulp.task('html', ['images'], () => {
  return gulp.src('src/html/**/*.html')
    .pipe(plumber({ errorHandler: onError }))
    .pipe(include({ prefix: '@', basepath: 'src/' }))
    .pipe(htmlmin({ collapseWhitespace: true, removeComments: false }))
    .pipe(size(sizes))
    .pipe(gulp.dest('dist'))
})
```

### SASS

Sass is a stylesheet language that’s compiled to CSS. It allows variables, nested rules, mixins, functions, and more, all with a fully CSS-compatible syntax. Sass helps keep large stylesheets well-organized and makes it easy to share design within and across projects. Because that SASS is not natively supported by most of the browsers, before building into production, we need to compile the SASS code into native CSS.

The compiling is processed in a Gulp pipeline implemented as below:

```js
// sass

const processors = [
  rucksack({ inputPseudo: false, quantityQueries: false }),
  prefixer({ browsers: 'last 2 versions' }),
  cssnano({ safe: true })
]

gulp.task('sass', () => {
  return gulp.src('src/sass/style.scss')
    .pipe(plumber({ errorHandler: onError }))
    .pipe(maps.init())
    .pipe(sass())
    .pipe(postcss(processors))
    .pipe(size(sizes))
    .pipe(maps.write('./maps', { addComment: false }))
    .pipe(gulp.dest('dist'))
})
```

### ECMAScript6

ECMAScript6 is the sixth edition of ECMAScript language specification standard which is used in the implementation of JavaScript. Similar to SASS, codings in this language need to be compiled into an earlier version of JavaScript to avoid potential compatibility issues.

The compiling is processed in a Gulp pipeline implemented as below:

```js
const read = {
  input: 'src/js/main.js',
  output: {
    sourcemap: true
  },
  plugins: [
    resolve({ jsnext: true, main: true }),
    commonjs(),
    babel({
      babelrc: false,
      presets: [
        [
          '@babel/preset-env', {
            modules: false,
            targets: {
              browsers: ['last 2 versions']
            }
          }
        ]
      ],
      plugins: [

      ]
    }),
    uglify(),
    filesize()
  ]
}
```



## Integration

The frontend-backend integration process encountered problems of CORS/CORB. In order to eliminate safety controls during the local development testing, a copy of Chrome configuration is needed:

```
"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe" --disable-web-security --disable-gpu --user-data-dir=~/chromeTemp
```





# IV. Deployment and Reporting



## Deployment

In order for the project to be accessible online and closer to the production environment during the testing process, we need to deploy the front end and back end of the entire project to a remote server for testing and access.

Using the 2GB memory server that AWS deployed in Paris, our deployment process went smoothly—every command will get a quick response and run results.



![1.png](https://i.loli.net/2019/06/12/5d00404d7d4b665051.png)



*Fig - Instance from AWS*

We used the latest LTS version (18.04) of Ubuntu system as the operating system, Ubuntu is a free and open-source Linux distribution based on Debian. Because it is really suitable for our project. The system's own Python3 environment is exactly what we need, and the powerful apt tool allows us to easily install the latest versions of the various tools we need, such as pip3 / NodeJs / Npm / Nginx / Jenkins.

![3.png](https://i.loli.net/2019/06/12/5d00404d8915920286.png)

*Fig - Some of the environments*

Pip is a package-management system used to install and manage software packages written in Python. Many packages can be found in the default source for packages and their dependencies — Python Package Index (PyPI).

With pip, we can install a required Python environment with a single command at a time, such as Tensorflow / Pandas / NumPy / Flask, so that our online environment will be as good as the development environment.

Node.js is an open-source, cross-platform JavaScript run-time environment that executes JavaScript code outside of a browser. Node.js lets developers use JavaScript to write command line tools and for server-side scripting—running scripts server-side To produce dynamic web pagecontent before the page is sent to the user's web browser.

NodeJs is used because the front-end project uses this environment. It not only provides the Npm package management tool, but also provides a running environment for many useful tools, such as gulp used in this project, which can provide automatic image compression and CSS format conversion. And other practical features.

Nginx is a web server which can also be used as a reverse proxy, load balancer, mail proxy and HTTP cache.

With Nginx, we can use reverse proxy to put the front and back projects deployed on different ports and different directories under a unified domain name. This avoids the server opening more ports to the outside world, enhances security, and by the way, solves the Cross-domain issues.

![2.png](https://i.loli.net/2019/06/12/5d00404d8915212001.png)

*Fig - Back-End API*

At the beginning, because there is always a new environment added, we manually clone or pull the code repository and enter commands to deploy each time. This is very cumbersome, but it is a must.

When the project is in the middle and late stages, the framework has been basically laid, and there will be no major changes. The main purpose of redeployment is to test the changes brought by a few code changes. If you manually deploy the project every time, it will lead to a lot of homogenization. And meaningless work, so at this time we deployed Jenkins on the server as a persistent integration application to avoid the huge workload caused by each manual deployment.



## Continuous integration (CI)

Jenkins is an open source automation server written in Java. Jenkins helps to automate the non-human part of the software development process, with continuous integration and facilitating technical aspects of continuous delivery. It is a server-based system that runs in servlet containers such as Apache Tomcat.
With Jenkins, we can finally omit the same deployment command every time, replace it with a web page that accesses Jenkins and click the Build button to complete the deployment.

![4.png](https://i.loli.net/2019/06/12/5d004a793c02337857.png)

*Fig - Deploy commands*

But this is not enough. It still requires us to visit the webpage to build the project when we want to redeploy the project. It does not achieve complete automation and brings unnecessary workload.
![5.png](https://i.loli.net/2019/06/12/5d004a79911fd68372.png)

*Fig - Jenkins build history*

So we use the GitHub plugin provided by Jenkins, in conjunction with the GitHub repository settings, to send a network request to the Jenkins server when the code repository changes - this is called Webhook, which allows the Jenkins server to know the code changes and build it automatically.
A webhook in web development is a method of augmenting or altering the behavior of a web page, or web application, with custom callbacks. These callbacks may be maintained, modified, and managed by third-party users and developers who may not even be affiliated With the originating website or application.
![6.png](https://i.loli.net/2019/06/12/5d004a79e575a72608.png)

*Fig - GitHub webhook settings*

In this way, complete automation is achieved. After the changed code is pushed to the remote code repository, all we need to do is wait for the deployment to complete. If the deployment succeeds, the new web page or API will be displayed under the corresponding domain name.

### Thoughts

Although our server's hardwares are pretty efficient as a student server, it still takes about 2.5 minutes for each deployment, which is not normal even for Jenkins, a server-critical tool. After carefully observing the console output during the build process, I found that the most time-consuming command was `npm install`, but because the backend developer did not include the files generated after the build when writing the gitignore file, each build would need to be The entire local repository is emptied and re-clone, so `npm install` cannot be omitted and should be improved in later development of the project.



## Software Testing





# V. Timeline & Work Repartition



## I. Initialization Step

### Tasks

1. Study the problem context by choosing the data you want to mine.
2. Elaborate the Use-case diagram and detailed description of the most important cases.
3. Define the global architecture of the Project.

### Timeline & Work Repartition

| Task ID | Collaborators            | Estimated Schedule | Status |
| ------- | ------------------------ | ------------------ | ------ |
| I-1     | All members              | Apr 29 - May 01    | ✅      |
| I-2     | Chunyang, Huanyu, Huimin | Apr 29 - May 06    | ✅      |
| I-3     | Jingtao                  | May 01 - May 03    | ✅      |

### Risks & Difficulties

- Nonproficiency in creating UML diagrams



## II. Elaboration step

### Given Tasks

1. Detailed architecture of the Project by describing all the functionalities and the employed languages.
2. Scraping and collect the data.
3. Data cleaning and transformation.
4. Analysis of the dataset.
   - a. Analysis of the GeoNames dataset.
   - b. Analysis of the GeoNames `JOIN` GapMinder Countries dataset.

### Additional Tasks

5. Graphical user interface (GUI) prototyping & design.
6. API documents drafting.
7. GUI Frontend implementation (Static).

### Timeline & Work Repartition

| Task ID | Collaborators            | Estimated Schedule | Status |
| ------- | ------------------------ | ------------------ | ------ |
| II-1    | Huanyu, Chunyang         | May 06 - May 07    | ✅      |
| II-2    | All members              | May 07             | ✅      |
| II-3    | All members              | May 07 - May 09    | ✅      |
| II-4a   | Jingtao                  | May 09 - May 14    | ✅      |
| II-4b   | Huanyu, Chunyang, Huimin | May 09 - May 14    | ✅      |
| II-5    | Jingtao                  | May 06 - May 10    | ✅      |
| II-6    | Huanyu, Chunyang         | May 10 - May 14    | ✅      |
| II-7    | Jingtao                  | May 10 - May 15    | ✅      |



## III. Construction step

### Given Tasks

1. Integration of all the cases defined in the elaboration step.
2. Machine Learning.
   - a. Clustering. Candidate models: K-Means, DBScan, SpectralClustering.
   - b. Classification Model Training. Model: Recurrent Neural Network.
   - c. Regression Model Training. Which model to use is still under research, but likely to be Recurrent Neural Network as well. We're still dealing with potential issues brought by the non-Cartesian output values (global coordinates).

3. Visualization of the dataset.
4. Program the application and make the main tests.
   - a. Backend programming - Database & API Implementation.
   - b. Frontend-backend docking.
   - c. GUI Frontend completion (Interactive).

### Additional Tasks

5. Model & parameters optimization (If applicable).

### Timeline & Work Repartition

| Task ID | Collaborators    | Estimated Schedule | Status |
| ------- | ---------------- | ------------------ | ------ |
| III-1   | -                | -                  | ✅      |
| III-2a  | Jingtao, Huanyu  | May 13 - May 19    | ✅      |
| III-2b  | Jingtao, Huanyu  | May 13 - May 19    | ✅      |
| III-2c  | All members      | May 19 - May 26    | ✅      |
| III-3   | Jingtao          | May 15 - May 20    | ✅      |
| III-4a  | Huanyu, Chunyang | May 20 - May 24    | ✅      |
| III-4b  | All members      | May 24 - May 27    | ✅      |
| III-4c  | Jingtao          | May 24 - May 27    | ✅      |
| III-5   | All members      | May 20 - May 25    | ✅      |



## IV. Deployment and Reporting

### Given Tasks

1. Deploy the project if possible in the defined environment
2. Prepare a detailed report
3. Presentation

### Additional Tasks

4. Continuous integration (CI) deployment

### Timeline & Work Repartition

| Task ID | Collaborators | Estimated Schedule | Status |
| ------- | ------------- | ------------------ | ------ |
| IIII-1  | Huimin        | May 27 - May 29    | ✅      |
| IIII-2  | All members   | May 29 - June 14   | ✅      |
| IIII-3  | All members   | May 29 - June 05   |        |
| IIII-4  | Huimin        | May 29 - June 04   | ✅      |





# VI. Conclusion

Place names around the world have a subtle and close relationship with their location. This is usually because different regions have different languages and writing systems, hence the different spelling patterns.

Characteristics of place names differ due to geographic location changes. To a certain extent, the changes are both continuous and discrete. They're continuous because place names are similar in neighboring regions. For instance, throughout the Europe continent, the proportion of consonants in their spelling rises from the Mediterranean to the north. Another example, inside China, the unified country, the place names in the northwest are more "Arabic" than those in the east. 

The discrete character of place names, on the other hand, is usually caused by geographical barriers, political boundaries, and cultural divisions. For example, in Spain and Argentina, two countries that locates vastly different, have similar place names because of historical colonial activities. Like the border between China and Vietnam, they share similar cultures, but the official language on one side is In Chinese, while Vietnamese on the other side - the two writing systems, therefore, have different Latin transcription standards, resulting in the latter's place names usually being split into many relatively short words. This "discrete" character is caused by political reasons.

Some geographical regions have quite obvious place name patterns. For instance, in East Europe, especially Russia, place names are usually constructed by a single long word, which makes them relatively easier to be recognized. Some Slavic suffixes like “-sk” are commonly seen here. Some geographical areas, such as sub-Saharan Africa and Oceania island countries, the culture and languages are so diverse there causing the place names are much more challenging to identify.