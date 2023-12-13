# Takes in raw ticker data and creates the technical indicators to feed into Framework.py
# Also models the residuals for the O_U file by creating new columns

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import TimeSeriesSplit
import pickle
from O_U import OU

#---------------------------------------------------------------------------------------#

# Before creating our trading signals, we first must pull in our target raw data from
# the data dump for modification:

# Get in the number of classes and their ticker names from user
# Open up the data files that the user specified

numClasses = 0
classA = " "
classB = " "

# Get the user's path to the data folder we want to read from
temp_path = os.path.abspath(os.path.dirname('path.txt'))
idx = temp_path.find('Stock-Market-Trading-Bot')
path = temp_path[:(idx+24)]
pathname = path + '/Source-Code/Backend/Data/Raw_Dump/'

stockName = input("Name for this stock: ")

while (numClasses < 1 or numClasses > 2):
    numClasses = int(input("Number of classes for this stock (1 or 2): "))
    if (numClasses == 1):
        classA = input("Ticker name for the stock: ")
        filename = pathname + classA + '.csv'
        data1 = pd.read_csv(filename).iloc[:, 1:]
        print("Data loaded for: " + classA)
        # Note: If only 1 class, we need another dataset to compare this class to for 
        # the SVM model to work properly
    elif (numClasses == 2):
        classA = input("Ticker name for the first class: ")
        filename = pathname + classA + '.csv'
        data1 = pd.read_csv(filename).iloc[:, 1:]
        classB = input("Ticker name for the second class: ")
        filename = pathname + classB + '.csv'
        data2 = pd.read_csv(filename).iloc[:, 1:]
        print("Data loaded for: " + classA)
        print("Data loaded for: " + classB)

#---------------------------------------------------------------------------------------#

# Now it is time to create the trading signals so our bot will know when to trade:

#---------------------------------------------------------------------------------------#

# Our first trading signal will be a Simple Moving Average:

# This function uses close prices calculated within a window range. We will then
# keep track of the moving average and return this value

def sma(price_data, window):
    sma = price_data.rolling(window).mean()[window - 1:]
    sma.index += 1
    sma = sma[:-1]
    return sma

#---------------------------------------------------------------------------------------#

# Our next trading signal will be a Bollinger Band:

# This is one of the most used algorithms in trading due to it's simplicity and effectiveness
# We need 3 bands: an upper, lower, and middle band. The upper and lower bands are plotted
# two standard deviations away from the average close price. Due to these deviations,
# these two bands will hold over 80% of the price action, making anything that falls outside
# this range very significant.

# We want to buy when the the close prices reaches the lower band and sell when the close price
# reaches the upper band.

def BB(data):
    # Implementation needed
    return data

 
#---------------------------------------------------------------------------------------#

# Our next trading signal will be a Relative Strength Index:

# This function will measure the speed and change of price movements and will be very important
# in determining when to buy/sell one of the classes.

def rsi(data, window):
    i = 1
    pos_period = [0]
    neg_period = [0]
    dataOpen = data['OPEN']

    while i < dataOpen.index[-1]:
        if dataOpen[i] > dataOpen[i - 1]:
            pos_period.append(dataOpen[i])
            neg_period.append(0)
        else:
            pos_period.append(0)
            neg_period.append(dataOpen[i])
        i += 1
    
    pos_period = pd.Series(pos_period)
    neg_period = pd.Series(neg_period)

    pos_sum = pd.Series(pos_period.rolling(window).sum())
    neg_sum = pd.Series(neg_period.rolling(window).sum())

    temp = (window - pos_sum) / (window - neg_sum)
    rsi = abs(100 - (100 / (1 + temp)))
    return rsi[window:]

#---------------------------------------------------------------------------------------#

# Our final trading signal will be a Money Flow Index:

# We must use some sort of trading signal which tracks the volume across our data classes.

# This function will look at the Money Flow Index for a certain index and compare it to 
# that of the index behind it. This function is fairly simple and tracks volume along
# with the close prices when available.

def mfi(data, window):
    money_flow = (data['HIGH'] + data['LOW'] + data['CLOSE']) / 3
    positives = [0, 0]
    negatives = [0, 0]
    i = 1

    while i < data.index[-1]:
        if money_flow[i] > money_flow[i - 1]:
            positives.append(money_flow[i] * data.loc[i, 'VOLUME'])
            negatives.append(0)
        else:
            positives.append(0)
            negatives.append(money_flow[i] * data.loc[i, 'VOLUME'])
        i += 1

    positives = pd.Series(positives)
    negatives = pd.Series(negatives)

    positive_sum = pd.Series(positives.rolling(window).sum())
    negative_sum = pd.Series(negatives.rolling(window).sum())
    
    mfi = (window - positive_sum) / (window - negative_sum)
    mfi = abs(100 - (100 / (1 + mfi)))

    return mfi[window:]

#---------------------------------------------------------------------------------------#

# Now all of our trading signals have been defined

# We will next use all of these trading signals to modify our raw data to include
# the necessary technical indicators that will be fed into O_U.py:

# Range will be our window size. We will set it to 10 for testing

# Get the user's path to the folder we want to save the processed data to
temp_path = os.path.abspath(os.path.dirname('path2.txt'))
idx = temp_path.find('Stock-Market-Trading-Bot')
path = temp_path[:(idx+24)]
pathname = path + '/Source-Code/Backend/Data/Processed_Dump/'

range = 10
if (numClasses == 2):
    data1['sma'] = sma(data1['CLOSE'], range).pct_change()
    data2['sma'] = sma(data2['CLOSE'], range).pct_change()

    #data1['BB'] = BB(data1).pct_change()
    #data2['BB'] = BB(data2).pct_change()

    data1['rsi'] = rsi(data1, range).pct_change()
    data2['rsi'] = rsi(data2, range).pct_change()

    data1['mfi'] = mfi(data1, range).pct_change()
    data2['mfi'] = mfi(data2, range).pct_change()

    data1['price'] = data1['CLOSE'].pct_change()
    data2['price'] = data2['CLOSE'].pct_change()

    processed_data1 = data1[range+1:].reset_index(drop = True)
    processed_data2 = data2[range+1:].reset_index(drop = True)

    processed_data1.to_csv(pathname + classA + '_processed.csv')
    print("Data processed for: " + classA)
    processed_data2.to_csv(pathname + classB + '_processed.csv')
    print("Data processed for: " + classB)

elif (numClasses == 1):
    data1['sma'] = sma(data1['CLOSE'], range).pct_change()
    #data1['BB'] = BB(data1).pct_change()
    data1['rsi'] = rsi(data1).pct_change()
    data1['mfi'] = mfi(data1, range).pct_change()
    data1['price'] = data1['CLOSE'].pct_change()

    processed_data1 = data1[range+1:].reset_index(drop=True)

    processed_data1.to_csv(pathname + classA + '_processed.csv')
    print("Data processed for: " + classA)

#---------------------------------------------------------------------------------------#

# Finally, we will model our residuals using the O-U model and feed this into our O_U file:

# We want to create a new column in our data with the value either 0 or 1. The value will be 
# 0 if spread of residuals is within our threshold and 1 if the spred exceeds our threshold.
# We will use a threshold of 0.001 and spread of 10

def create_list(threshold = 0.001): 
    def addResiduals(spread):
        min = spread[::-1].rolling(window = 10).min()[::-1]
        min.iloc[-10:] = spread.iloc[-10:]

        zero_or_one = (spread - min) > threshold # If outside threshold: 1
        val = zero_or_one.astype(int) # Convert our bool to either 0 or 1

        return val
    return addResiduals

# Create our info 4d dataframe to be used for Framework.py
InfoPath = path + '/Source-Code/Backend/Data/Info/'
list = create_list(threshold = 0.001)
New_OU = OU(processed_data1, processed_data2)
New_OU.slide(m_size = 2000, e_size = 100)
info = New_OU.split(['price', 'sma', 'rsi', 'mfi'], labels = list, weight = True)
np.save(InfoPath + stockName + '_info.npy', info)
print("Info file generated for " + stockName)