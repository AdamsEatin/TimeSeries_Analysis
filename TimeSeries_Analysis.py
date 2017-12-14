# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 02:37:57 2017

@author: Adam
"""

import scipy
import numpy
import matplotlib
import pandas 
import sklearn
import statsmodels
import warnings

from scipy.stats import boxcox
from matplotlib import pyplot
from pandas import Series, DataFrame
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima_model import ARIMA, ARIMAResults
from sklearn.metrics import mean_squared_error
from math import sqrt


def split_dataset(data):
    split_point = len(data) - 10
    dataset = data[0:split_point]
    validation = data[split_point:]
    dataset.to_csv('dataset.csv')
    validation.to_csv('validation.csv')
    return

#Evaluate arima model for given order and return RMSE value
def evaluate_arima_model(X, arima_order):
    #Preparing the training dataset
    X = X.astype('float32')
    
    train_size = int(len(X) * 0.50)
    train = X[0:train_size]
    test = X[train_size:]
    history = [x for x in train]
    
    #Making predictions
    predictions = list()
    for t in range(len(test)):
        model = ARIMA(history, order=arima_order)
        
        #model_fit - model.fit(disp=0)
        model_fit = model.fit(trend='nc', disp=0)
        yhat = model_fit.forecast()[0]
        predictions.append(yhat)
        history.append(test[t])
    
    #Calculate out sample error
    mse = mean_squared_error(test, predictions)
    rmse = sqrt(mse)
    return rmse

#Evaluate combinations of p, d, q values for the ARIMA model
def evaluate_models(dataset, p_values, d_values, q_values):
    dataset = dataset.astype('float32')
    best_score = float("inf")
    best_cfg = None
    
    for p in p_values:
        for d in d_values:
            for q in q_values:
                order = (p, d, q)
                try:
                    mse = evaluate_arima_model(dataset, order)
                    if mse < best_score:
                        best_score = mse
                        best_cfg = order
                    print('ARIMA ',order,' RMSE= ',mse)
                except:
                    continue
    print('Best Config: ',best_cfg, ' RMSE: ',best_score)
    return best_cfg
   
#Evaluate Residual Error Bias
def evaluate_bias(X, arima_order):
    #Preparing the training dataset
    X = X.astype('float32')
    
    train_size = int(len(X) * 0.50)
    train = X[0:train_size]
    test = X[train_size:]
    
    #walk-forward validation
    history = [x for x in train]
    predictions = list()
    
    for t in range(len(test)):
        model = ARIMA(history, order=arima_order)
        model_fit = model.fit(trend='nc', disp=0)
        yhat = model_fit.forecast()[0]
        predictions.append(yhat)
        
        #Observation
        obs = test[t]
        history.append(obs)
    
    #Errors
    residuals = [test[i]-predictions[i] for i in range(len(test))]
    residuals = DataFrame(residuals)
    residuals_mean = residuals.mean().iloc[0]
    
    return residuals_mean

#patch around bug in ARIMA class
def __getnewargs__(self):
	return ((self.endog),(self.k_lags, self.k_diff, self.k_ma))
 
ARIMA.__getnewargs__ = __getnewargs__

def generate_model(X, arima_Order, bias):
    X = X.astype('float32')
    
    #fit model
    model = ARIMA(X, arima_Order)
    model_fit = model.fit(trend='nc', disp=0)
    
    #save model
    model_fit.save('model.pkl')
    numpy.save('model_bias.npy', [bias])
    return
    

def validate_model(X, Y, arima_Order):
    X = X.values.astype('float32')
    Y = Y.values.astype('float32')
    history = [x for x in X]
    
    #load model
    model_fit = ARIMAResults.load('model.pkl')
    bias = numpy.load('model_bias.npy')
    
    #Make first prediction
    predictions = list()
    yhat = bias +  float(model_fit.forecast()[0])
    predictions.append(yhat)
    history.append(Y[0])
    print('\nPrediction: ',yhat,' Expected: ',Y[0])
    
    #Rolling forecasts
    for i in range(1,len(Y)):
        #Make predictions
        model = ARIMA(history, arima_Order)
        model_fit = model.fit(trend='nc', disp=0)
        yhat = bias + float(model_fit.forecast()[0])
        predictions.append(yhat)
        
        #Observations
        obs = Y[i]
        history.append(obs)
        print('Prediction: ',yhat,' Expected: ',obs)
        
    #Reporting Performance
    mse = mean_squared_error(Y, predictions)
    rmse = sqrt(mse)
    
    pyplot.plot(Y)
    pyplot.plot(predictions, color='red')
    pyplot.show()
    print('\nRMSE: ',rmse)
    return

#Generate a prediction for the next year
#TODO: ALlow for recursion to generate predictions for multiple years
def generate_prediction(X, arima_Order):
    X = X.values.astype('float32')
    xList = X.tolist()
    
    model_fit = ARIMAResults.load('model.pkl')
    bias = numpy.load('model_bias.npy')
    
    predictions = list()
    yhat = bias + float(model_fit.forecast()[0])
    predictions.append(yhat)
    print(predictions)
    
    
    #xList = X.tolist()
    #xList.append(420.0)
    return

#test function for recursive predictions
def recursive_test(data, arima_Order, numOfPredictions):
    
    def make_predictions(xList, yList, model, bias, numOfPredictions):
        count = 0
        
        while count < numOfPredictions:
            model = ARIMA(xList, arima_Order)
            model_fit = model.fit(trend = 'nc', disp=0)
            
            prdct = bias + float(model_fit.forecast()[0])
            
            xList.append(prdct[0])
            print("Prediction ",count," : ",prdct[0])
            yList.append(prdct[0])
            
            count = count + 1
        
        return yList
    
    X = data.values.astype('float32')
    xList = X.tolist()
    yList = []
    model_fit = ARIMAResults.load('model.pkl')
    bias = numpy.load('model_bias.npy')
    
    predictions = make_predictions(xList, yList, model_fit, bias, numOfPredictions)
    
    print("\n xList:")
    print(xList)
    print("\n yList:")
    print(yList)
    
    pyplot.plot(xList, color = 'blue')
    pyplot.plot(predictions, color = 'red')
    pyplot.show()
    
    return


#Original Dataset  
data = Series.from_csv('SampleData1.csv', header=0)

#Training Dataset
X = Series.from_csv('dataset.csv')

#Validation Dataset
Y = Series.from_csv('validation.csv')

#Number of predictions to be made
numOfPredictions = 4

p_Values = range(0,5)
d_Values = range(0,3)
q_Values = range(0,5)
warnings.filterwarnings("ignore")

arima_Order = evaluate_models(X.values, p_Values, d_Values, q_Values)

bias = evaluate_bias(X, arima_Order)

generate_model(X, arima_Order, bias)

#validate_model(X, Y, arima_Order)

#generate_prediction(X, arima_Order)

recursive_test(X, arima_Order, numOfPredictions)