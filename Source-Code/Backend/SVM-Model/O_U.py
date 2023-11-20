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
