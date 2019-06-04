## P&I Backend API Document

+ Update Log
  
  - 2019.6.3
  
    1.Edit API Document
  
  ​       2.Add Three API :[Predict city's area](#PredictCity), [Country statistics](#CountryStatistics), [Country list](#CountryList)
  
  - 2019.6.4  
  
    1.Add Update Log
  
    2.Change the resonse of [Predict city's area](#PredictCity)
  
    3.Fix some bugs

### <span id="PredictCity">Predicting city's area</span>

  - `###### POST /city/predict`

- parameter

 <table>
    <tr>
        <th>Name</th>
        <th>Type</th>
        <th>Description</th>
    </tr>
    <tr>
        <td>city</td>
        <td>string</td>
        <td>inputting city's name</td>
    </tr>
</table>  

- response

```javascript
{
    "code": 200,
    "area": [
        [
            -0.9534,
            "EastAsia"
        ],
        [
            -1.3968,
            "WEurope"
        ],
        [
            -2.4532,
            "EEurope"
        ]
    ]
}
```

<font size=2 color=grey>areas are in 'EastAsia', 'S&SEAsia', 'EnUsAuNz', 'Latinos', 'Arabics', 'WEurope', 'EEurope', 'Oceania', 'SSAfrica'</font>

--------------------

### Statistic country's attributes

- `###### POST /country/statistic`

- parameter

<table>
    <tr>
        <th>Name</th>
        <th>Type</th>
        <th>Description</th>
    </tr>
    <tr>
        <td>country</td>
        <td>string</td>
        <td>inputting country's name</td>
    </tr>
</table> 

- response

```javascript
{
    "country": "China",
    "code": 200,
    "energy_production": {
        "2001": 0.000855948,
        "2002": 0.000910788,
        "2003": 0.0010184760000000001,
        "2004": 0.0011482110000000001,
        "2005": 0.001241143,
        "2006": 0.0013146820000000002,
        "2007": 0.001380643,
        "2008": 0.001497995,
        "2009": 0.0015618589999999999,
        "2010": 0.0015618589999999999
    },
    "tax": {
        "2001": 7.402022108,
        "2002": 8.503401634,
        "2003": 8.538996219,
        "2004": 8.860551515,
        "2005": 8.679594442,
        "2006": 9.188060347,
        "2007": 9.928816567,
        "2008": 10.26866091,
        "2009": 10.53972933,
        "2010": 10.53972933
    },
  ...
}
```
Last ten years or less

--------------

### Country list

  - `###### GET /country/list`
  - response
```javascript
{
    "names": [
        "Andorra",
        "United Arab Emirates",
        "Afghanistan",
        "Antigua and Barbuda",
        "Anguilla",
        "Albania",
        "Armenia",
        "Netherlands Antilles",
        "Angola",
      ....
      ]
}
```
