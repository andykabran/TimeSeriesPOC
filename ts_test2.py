# -*- coding: utf-8 -*-
"""
Created on Fri Aug  6 09:57:53 2021

@author: andyk
"""

from fbprophet import Prophet
from pandas import read_csv
from pandas import to_datetime
# from matplotlib import pyplot as plt
from pandas import DataFrame

from flask import Flask
from flask import jsonify

app = Flask(__name__)


# import fbprophet
# print('Prophet %s' % fbprophet.__version__) #print version number

@app.route('/')
def get_predict():
    # load and plot car sales dataset
    path = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/monthly-car-sales.csv'

    # path = open(r'C:\Users\andyk\Desktop\monthly-car-sales.csv','r')

    df = read_csv(path, header=0)
    print(df.shape)
    #
    # prepare expected column
    df.columns = ['ds', 'y']
    df['ds'] = to_datetime(df['ds'])

    # define the model
    model = Prophet()

    # fit the model
    model.fit(df)

    # define a model for which we want a prediction
    future = list()
    for i in range(1, 13):
        date = '1969-%02d' % i
        future.append([date])
    future = DataFrame(future)
    future.columns = ['ds']
    future['ds'] = to_datetime(future['ds'])

    # use the model to make a forecast
    forecast = model.predict(future)

    # summarize the forecast

    print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].head())
    # print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']])

    forecast_table = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
    forecast_json = forecast_table.to_json()
    print(forecast_table.head())
    print(forecast_json)

    # plot forecast
    # model.plot(forecast)
    # plt.show()

    return jsonify(forecast_table)
