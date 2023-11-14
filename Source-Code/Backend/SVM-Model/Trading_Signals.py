# Takes in raw ticker data and creates the technical indicators to feed into Framework.py
# Also models the residuals for the O_U file by creating new columns

import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt

# from sklearn import preprocessing
# from sklearn.model_selection import TimeSeriesSplit

import pickle

#from O_U import OU

#---------------------------------------------------------------------------------------#

# Get in the number of classes and their ticker names from user
# Open up the data files that the user specified

numClasses = 0
classA = " "
classB = " "
while (numClasses < 1 or numClasses > 2):
    numClasses = int(input("Number of classes for this stock (1 or 2): "))
    if (numClasses == 1):
        classA = input("Ticker name for the stock: ")
        data1 = pd.read_csv('/Users/dominic/Desktop/CEN_Project/ML-Trading-Bot/Source-Code/Backend/Data/Dump/' + classA + '.csv').iloc[:, 1:]
        print("Data loaded for: " + classA)
        # Note: If only 1 class, we need another dataset to compare this class to for 
        # the SVM model to work properly
    elif (numClasses == 2):
        classA = input("Ticker name for the first class: ")
        classB = input("Ticker name for the second class: ")
        data1 = pd.read_csv('/Users/dominic/Desktop/CEN_Project/ML-Trading-Bot/Source-Code/Backend/Data/Dump/' + classA + '.csv').iloc[:, 1:]
        data2 = pd.read_csv('/Users/dominic/Desktop/CEN_Project/ML-Trading-Bot/Source-Code/Backend/Data/Dump/' + classB + '.csv').iloc[:, 1:]
        print("Data loaded for: " + classA)
        print("Data loaded for: " + classB)

#---------------------------------------------------------------------------------------#

# Now it is time to create the trading signals so our bot will know when to trade

#---------------------------------------------------------------------------------------#

# Our first trading signal will be a simple moving average:

# We will need to create 2 different functions for this signal

# This first function uses close prices calculated within a window range. Note: this will
# not use the current period's data since we will only have that once we reach the end of 
# the minute and by then we will be in the next window

def smaClose(prices, window):
    sma = prices.rolling(window).mean()[window-1:]
    sma.index += 1
    sma = sma[:-1]
    return sma

# This next function finishes what the first function could not by using open prices

def smaOpen(prices, window):
    sma = prices.rolling(window).mean()[window-1:]
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

# def BB(data):

 
#---------------------------------------------------------------------------------------#

# Our next trading signal will be a Moving Average Convergence Divergence Indicator:


#---------------------------------------------------------------------------------------#

# Our next trading signal will be a Change of Character:


#---------------------------------------------------------------------------------------#

# Now all of our trading signals have been defined

# We will next use all of these trading signals to modify our raw data to include
# the necessary technical indicators that will be fed into O_U.py

# Range will be our window size. We will set it to 10 for testing

range = 10
if (numClasses == 2):
    data1['sma'] = smaClose(data1['CLOSE'], range).pct_change()
    data2['sma'] = smaClose(data2['CLOSE'], range).pct_change()

    processed_data1 = data1[range+1:].reset_index(drop=True)
    processed_data2 = data2[range+1:].reset_index(drop=True)

    processed_data1.to_csv(classA + '_processed.csv')
    processed_data2.to_csv(classB + '_processed.csv')

elif (numClasses == 1):
    data1['sma'] = smaClose(data1['CLOSE'], range).pct_change()

    processed_data1 = data1[range+1:].reset_index(drop=True)

    processed_data1.to_csv(classA + '_processed.csv')

#---------------------------------------------------------------------------------------#

# Finally, we will model our residuals using the O-U model and feed this into our O_U file

# We want to create a new column in our data with the value either 0 or 1. The value will be 
# 0 if spread of residuals is within our threshold and 1 if the spred exceeds our threshold.
# We will use a threshold of 0.001 and spread of 10

def create_list(): 
    def addResiduals(spread):
        min = spread[::-1].rolling(window = 10).min()[::-1]
        min.iloc[-10:] = spread.iloc[-10:]

        zero_or_one = (spread - min) > 0.001 # If outside threshold: 1
        val = int(zero_or_one == True) # Convert our bool to either 0 or 1

        return val
    return create_list

list = create_list()
New_OU = OU(processed_data1, processed_data2)
New_OU.split_slide(m_size=2000, e_size=100)
