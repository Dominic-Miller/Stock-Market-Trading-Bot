# Takes in raw ticker data and creates the technical indicators to feed into O-U.py

import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt

# from sklearn import preprocessing
# from sklearn.model_selection import TimeSeriesSplit

import pickle

# from O-U import O-U

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
        data1 = pd.read_csv('./Data/' + classA + '.csv').iloc[:, 1:]
        print("Data loaded for: " + classA)
        # Note: If only 1 class, we need another dataset to compare this class to for 
        # the SVM model to work properly
    elif (numClasses == 2):
        classA = input("Ticker name for the first class: ")
        classB = input("Ticker name for the second class: ")
        data1 = pd.read_csv('./Data/' + classA + '.csv').iloc[:, 1:]
        data2 = pd.read_csv('./Data/' + classB + '.csv').iloc[:, 1:]
        print("Data loaded for: " + classA)
        print("\nData loaded for: " + classB)

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

def smaOPEN(prices, window):
    sma = prices.rolling(window).mean()[window-1:]
    return sma

#---------------------------------------------------------------------------------------#

