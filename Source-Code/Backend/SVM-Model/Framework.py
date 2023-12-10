# Take in the processed ticker data from Trading_Signals.py and training windows from O_U.py
# Trains the Model and creates a backtesting framework

import pandas as pd
import numpy as np
import os
import math
import scipy.stats
import sklearn
from sklearn import preprocessing
from sklearn import svm
import pickle

#---------------------------------------------------------------------------------------#

# The first thing we need to do is to pull in our processed data on our target stocks:

# Get the user's path to the folder we want to read the data from
temp_path = os.path.abspath(os.path.dirname('path2.txt'))
idx = temp_path.find('ML-Trading-Bot')
path = temp_path[:(idx+14)]
pathname = path + '/Source-Code/Backend/Data/Processed_Dump/'

# Take in the two ticker names for the two stocks and load in the processed data
classA = input("Ticker name for the first class: ")
filename = pathname + classA + '.csv'
data1 = pd.read_csv(filename).iloc[:, 1:]
classB = input("Ticker name for the second class: ")
filename = pathname + classB + '.csv'
data2 = pd.read_csv(filename).iloc[:, 1:]
print("Data loaded for: " + classA)
print("Data loaded for: " + classB)

# Load in info data
info = np.load()

#---------------------------------------------------------------------------------------#

# The next thing we will need to do is create a dataframe where we can display our profits
# and losses so we can track our net profit over time:

def profit_loss_dataframe(ticker1, ticker2, info):

    classA = ticker1['TICKER']
    classB = ticker2['TICKER']

    dataframe = pd.DataFrame()
    dataframe_labels = pd.Series()

    for i in info:
        idx = i['test']['index']
        residuals = i['test']['residuals_transform_price']
        beta = i['train']['beta_fit_price']
        df_temp = pd.concat([ticker1.loc[idx]['CLOSE'], beta * ticker2.loc[idx]['CLOSE'], ticker1.loc[idx]['price'],
                             beta * ticker2.loc[idx]['price'], i['test']['residuals_transform_price'], ticker1.loc[idx]['TIMESTAMP']], axis=1)
        datafram = dataframe.append(df_temp)
        dataframe_labels = dataframe_labels.append(i['test']['labels'])

    dataframe['label'] = dataframe_labels
    dataframe.columns = [classA, 'beta*' + classB, classA + '_return', 'beta*' + classB + '_return', 'residual', 'TIMESTAMP', 'label']

    # Find the profit or loss of the last trade
    dataframe['beta*' + classB + '_gains'] = dataframe['beta*' + classB] - (1 - dataframe['beta*' + classB + '_return']) * dataframe['beta*' + classB]
    dataframe[classA + '_gains'] = dataframe[classA] - (1 - dataframe[classA + '_return']) * dataframe[classA]
    dataframe['profit'] = dataframe['beta*' + classB + '_gains'] - dataframe[classA + '_gains']
    for i, item in enumerate(dataframe['TIMESTAMP']):
        dataframe.loc[i, 'TIMESTAMP'] = pd.to_datetime(item)
    
    return dataframe

#---------------------------------------------------------------------------------------#

# All we have left before training our bot is to define a few more functions which will
# modify our datasets to be exactly how we need it for training:

# This function will find an SVM that will work based on the parameters dictionary:
def find_svm(param_dict, info):

    labels = []

    for i in info:
        temp_svm = svm.SVC(**param_dict)
        temp_svm.fit(i['train']['df_scale'], i['train']['labels'])
        label = temp_svm.predict(i['test']['df_scale'])
        labels.append(label)

    return np.hstack(labels)

# This next function will format a dictionary of parameters into a single string that can 
# be written to a file:
def format_parameters(param_dict):
    param = ', '.join("{!s}-{!r}".format(key, val) for (key, val) in param_dict.items())
    param = param.replace("{", "")
    param = param.replace("}", "")
    param = param.replace("'", "")
    param = param.replace(",", "")
    param = param.replace(" ", "")
    param = param.replace(":", "")
    param = param.replace(".", "")
    param = param.strip()

    return param

# This next function will find the sharpe of our profit/loss dataframe:
def find_sharpe(df):
    
    return sharpe

# This next function will find the sortino ratio of our profit/loss dataframe:
def find_ratio(df):

    return ratio

# This final function will perform our profit/loss backtesting and give our prediction
# labels which will then be used to tell the bot if it is correct or not:
def backtesting(df, label, params):

    return result

#---------------------------------------------------------------------------------------#

# Now we will call these functions to create our dataframe and display our profits:

pl_df = profit_loss_dataframe(data1, data2, info)
