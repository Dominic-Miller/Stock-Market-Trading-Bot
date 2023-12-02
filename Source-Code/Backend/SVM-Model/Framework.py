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

#---------------------------------------------------------------------------------------#

# The next thing we will need to do is create a dataframe where we can display our profits
# and losses so we can track our net profit over time:


