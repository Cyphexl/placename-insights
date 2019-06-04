from flask import Flask, request
import json
from models.predict import predict
from models.calculate import statistic,countries

app = Flask(__name__)

@app.route('/city/predict',methods = ["POST"])
def predictCityArea():
    cityName = request.form.get('city')
    area = predict(cityName)
    result={'code':200,'area':area}
    return json.dumps(result)

@app.route('/country/statistic',methods=["POST"])
def statisticCountry():
    countryName=request.form.get('country')
    s = statistic(countryName)
    s['country']=countryName
    s['code']=200
    return json.dumps(s)

@app.route('/country/list',methods=["GET"])
def countryList():
    c=countries()
    return json.dumps({"names":c})

if __name__ == '__main__':
    app.run()
