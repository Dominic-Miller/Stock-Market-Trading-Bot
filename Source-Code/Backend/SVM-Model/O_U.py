# We will use Ornstein-Uhlenbeck process to model the residual term because the O-U process is a
# stochastic process such that the object modeled by the process always drifts towards its long-term mean.

# Take in raw and turn it into training and testing windows for the ML model. 
# Also properly model the price residuals

import pandas as pd
import numpy as np
import scipy.stats
import sklearn
from sklearn.model_selection import TimeSeriesSplit

# We will create an OU class and define our functions within this class to model the
# residuals and turn our data into training data

class OU(object):

    # Function to make sure the data sets have equal dimensions and initialize the new data
    def initialize(data, ds1, ds2, model_size = None, eval_size = None):

        data.ds1 = ds2
        data.ds2 = ds2
        data.final_ds = None
        data.m_size = model_size
        data.e_size = eval_size
        data.fts = []
        data.split_idx = []
        data.splits = []

        assert(ds1.shape == ds2.shape)

    # Function 1 to find split indices for expanding window cross-validation
    # Takes in the data and number of splits we want for cross-validation
    def split_expand(data, n_splits = 5):

        tscv = TimeSeriesSplit(n_splits = n_splits)
        data.split_idx = list(tscv.split(data.ds1))

    # Function 2 to find split indices for expanding window cross-validation
    # Takes in the data, and the size of the training and testing models we want for cross-validation
    def split_slide(data, m_size = 30000, e_size = 10000):

        splits = []
        end_ind = m_size
        cur_ind = 0

        assert(m_size < data.ds1.shape[0])

        while end_ind < data.ds1.shape[0]:
            # Find training indices
            train_ind = np.array(np.arange(cur_ind, end_ind))

            # If test indices for last test split is less than e_size, use remaining
            if (end_ind + e_size) < data.ds1.shape[0]:
                test_ind = np.array(np.arange(end_ind, (end_ind + e_size)))
            else:
                test_ind = np.array(np.arange(end_ind, data.ds1.shape[0]))

            splits.append((train_ind, test_ind))
            end_ind += e_size
            cur_ind += e_size

        data.split_idx = splits

    # Function that takes in the features of two different classes of a stock and calculates the 
    # residuals which will then be used to find the T-Score.


    # Functions that Use the OU Model to transform the target features and then calculate the 
    # residuals and get a T-Score again.


    # Function that gets the splits

