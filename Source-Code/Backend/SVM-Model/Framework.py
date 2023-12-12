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
idx = temp_path.find('Stock-Market-Trading-Bot')
path = temp_path[:(idx+24)]
pathname = path + '/Source-Code/Backend/Data/Processed_Dump/'

# Get the name of the stock
stockName = input("Name for this stock: ")

# Take in the two ticker names for the two stocks and load in the processed data
classA = input("Ticker name for the first class: ")
filename = pathname + classA + '_processed.csv'
data1 = pd.read_csv(filename).iloc[:, 1:]
classB = input("Ticker name for the second class: ")
filename = pathname + classB + '_processed.csv'
data2 = pd.read_csv(filename).iloc[:, 1:]
print("Data loaded for: " + classA)
print("Data loaded for: " + classB)

# Load in info data
infoPath = path + '/Source-Code/Backend/Data/Info/'
info = np.load(infoPath + stockName + '_info.npy', allow_pickle = True)

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
    days = {}
    count = 0
    for i, time in enumerate(df['TIMESTAMP']):
        time = pd.to_datetime(time)
        time = time.date()
        time = time.strftime('%m/%d/%Y')
        number = df.iloc[i]['profit_timeline']
        if time in days.keys():
            days[time] = number + days[time]
        else:
            count += 1
            days[time] = number
    df = pd.DataFrame.from_dict(days, orient='index')
    
    sharpe = (df.mean() / df.std()) * np.sqrt(252)
    return sharpe

# This next function is a quick function to find the precision:
def find_precision(labels, label):
    precision = labels[np.logical_and(labels == 1, label == 1)].shape[0]/labels[labels == 1].shape[0]
    return precision 

# This next function will find the sortino ratio of our profit/loss dataframe:
def find_ratio(df):
    days = {}
    count = 0
    for i , time in enumerate(df['TIMESTAMP']):
        time = pd.to_datetime(time)
        time = time.date()
        time = time.strftime('%m/%d/%Y')
        number = df.iloc[i]['profit_timeline']
        if time in days.keys():
            days[time] = number + days[time]
        else:
            count += 1
            days[time] = number

    df = pd.DataFrame.from_dict(days, orient='index')

    ratio = (df.mean() / df[df < 0].std()) * np.sqrt(252)
    return ratio

# This final function will perform our profit/loss backtesting and give our prediction
# labels which will then be used to tell the bot if it is correct or not:
def backtesting(df, label, params):

    results = {}

    param_str = format_parameters(params)
    profit = 0.0

    profit_timeline = []
    trade_timeline = []
    held_timeline = []
    data = []

    temp_df = df.copy()
    temp_df['label'] = label

    # Iterate through our dataframe
    for row in temp_df.iterrows():
        cur_profit = 0.0
        profit = row[1]['profit']
        residual = row[1]['residual']

        # Iterate through our trading data
        # We will use a window of 10 and threshold of 0.001
        for position in data:
            position['fresh'] += 1
            position['profit'] += profit
            if(position['residual'] - 0.001 >= residual) or position['fresh'] >= 10:
                cur_profit += position['profit']
                trade_timeline.append(position['profit'])
                held_timeline.append(position['fresh'])
                data.remove(position)
        profit_timeline.append(cur_profit)
        profit += cur_profit

        if row[1]['label'] == 1 and residual > 0:
            data.append({'profit': 0, 'residual': residual, 'fresh': 0})
        
    temp_df['profit_timeline'] = profit_timeline

    # Add all of our necessary collumns to our returning profit/loss dataframe
    results['total_profit'] = profit
    results['daily_profit_timeline'] = profit_timeline
    results['trade_profit_timeline'] = trade_timeline
    results['time_held_timeline'] = held_timeline
    results['trades_executed'] = len(trade_timeline)
    results['params'] = params
    results['precision'] = find_precision(temp_df['label'], df['label'])
    results['mean_profit_per_trade'] = np.mean(trade_timeline)
    results['sharpe'] = find_sharpe(temp_df)
    results['sortino'] = find_ratio(temp_df)

    return results

#---------------------------------------------------------------------------------------#

# Now we will call these functions to create our dataframe and display our profits:

pl_df = profit_loss_dataframe(data1, data2, info)
