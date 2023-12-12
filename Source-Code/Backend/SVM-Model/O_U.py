# We will use Ornstein-Uhlenbeck process to model the residual term because the O-U process is a
# stochastic process such that the object modeled by the process always drifts towards its long-term mean.

# Take in raw and turn it into training and testing windows for the ML model. 
# Also properly model the price residuals

import pandas as pd
import numpy as np
import scipy.stats
import sklearn
from sklearn.model_selection import TimeSeriesSplit

#---------------------------------------------------------------------------------------#

# We will create an OU class and define our functions within this class to model the
# residuals and turn our data into training data:

class OU(object):

    #---------------------------------------------------------------------------------------#

    # This first function will make sure the data sets have equal dimensions and initialize 
    # the new data:

    def __init__(data, ds1, ds2, model_size = None, eval_size = None):

        data.ds1 = ds2
        data.ds2 = ds2
        data.final_ds = None
        data.m_size = model_size
        data.e_size = eval_size
        data.fts = []
        data.split_idx = []
        data.splits = []

        assert(ds1.shape == ds2.shape)

    #---------------------------------------------------------------------------------------#

    # These next two functions will split our indices to be used for our training data:

    # Function 1 to find split indices for expanding window cross-validation:
    # Takes in the data and number of splits we want for cross-validation
    def expand(data, n_splits = 5):

        tscv = TimeSeriesSplit(n_splits = n_splits)
        data.split_idx = list(tscv.split(data.ds1))

    # Function 2 to find split indices for expanding window cross-validation:
    # Takes in the data, and the size of the training and testing models we want for cross-validation
    def slide(data, m_size = 30000, e_size = 10000):

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

    #---------------------------------------------------------------------------------------#

    # The next function takes in the features of two different classes of a stock and calculates 
    # the residuals which will then be used to find the T-Score:
    def featureFit(self, d1, d2, feature):

        d1 = d1[feature]
        d2 = d2[feature]
        
        # Estimate linear relationship using a linear regression
        beta, dx, _, _, _ = scipy.stats.linregress(d2, d1)

        # Calculate the residuals
        residuals = d1 - (d2 * beta)

        sum = np.cumsum(residuals)
        lag_price = sum.shift(1)

        # Perform lag-1 auto regression on the x_t and the lag
        b, a, _, _, _ = scipy.stats.linregress(lag_price.iloc[1:], sum.iloc[1:])

        # Calculate paramters to create a t-score
        mu = a / (1 - b)
        sigma = np.sqrt(np.var(sum))
        
        t_score = (sum - mu) / sigma
        t_score.name = feature

        # Return absolute value of t_score since we only care about the spread
        t_score = np.abs(t_score)

        return {'tscore_fit_' + feature: t_score, 'residuals_fit_' + feature: residuals,
                'beta_fit_' + feature: beta, 'dx_fit_' + feature: dx,
                'mu_fit_' + feature: mu, 'sigma_fit_' + feature: sigma,
                'fit_index_' + feature: np.array(d1.index)}

    #---------------------------------------------------------------------------------------#

    # These next two functions will use the OU Model to transform the target features and then
    # calculate the residuals and get a T-Score which is very important:

    # This first function transforms the target feature vector slices using the OU model parameters that
    # we get from the fit() method. We will take in the slice of the first ticker feature vector, the
    # slice of the second ticker feature vector, and a dictionary to the parameter values.
    def transform(self, ticker1, ticker2, feature, param_dict):

        beta = param_dict['beta_fit_' + feature]
        dx = param_dict['dx_fit_' + feature]
        mu = param_dict['mu_fit_' + feature]
        sigma = param_dict['sigma_fit_' + feature]
        
        feat1 = ticker1[feature]
        feat2 = ticker2[feature]

        residuals = feat1 - (feat2 * beta)

        x_t = np.cumsum(residuals)

        t_score = (x_t - mu) / sigma
        t_score = np.abs(t_score)
        t_score.name = feature

        return {'tscore_transform_' + feature: t_score, 'residuals_transform_' + feature: residuals,
                'transform_index_': np.array(ticker1.index)}

    # This next function takes in the features of the two different classes and calculates
    # the residuals. It then estimates the parameters for the OU equation and turns in into 
    # a t-score.
    def transformFit(self, ticker1, ticker2, t1, t2, OU_params, OU_features = None):

        fit_dicts = {}
        t_score_dicts = {}

        for feature in OU_params:
            temp_dict = self.featureFit(ticker1, ticker2, feature)
            fit_dicts.update(temp_dict)
            t_dict = self.transform(t1, t2, feature, fit_dicts)
            t_score_dicts.update(t_dict)

        training_data = pd.DataFrame([fit_dicts[f] for f in fit_dicts.keys() if 'tscore' in f]).transpose()
        testing_data = pd.DataFrame([t_score_dicts[t] for t in t_score_dicts.keys() if 'tscore' in t]).transpose()

        if OU_features:
            for feature in OU_features:
                training_data[feature + '1'] = ticker1[feature]
                training_data[feature + '2'] = ticker2[feature]
                testing_data[feature + '1'] = t1[feature]
                testing_data[feature + '2'] = t2[feature]

        return {'train': {'ds': training_data, **fit_dicts}, 'test': {'ds': testing_data, **t_score_dicts}}

    #---------------------------------------------------------------------------------------#

    # This last function will get the final splits which will then create the dataset to be 
    # sent over to Framework.py for training:

    # This will return a final list of all fit and transformed data
    def split(self, OU_params, OU_features = None, labels = None, weight = False):

        assert(self.split_idx)
        final_list = []

        # Transform our fits using the train and test datasets for each of the splits
        for train, test in self.split_idx:
            ds_train1 = self.ds1.loc[train]
            ds_train2 = self.ds2.loc[train]
            ds_test1 = self.ds1.loc[test]
            ds_test2 = self.ds2.loc[test]
            ft = self.transformFit(ds_train1, ds_train2, ds_test1, ds_test2, OU_params, OU_features)
            ft['train']['index'] = train
            ft['test']['index'] = test

            # Create the labels for our data
            if labels:
                training_labels = labels(ft['train']['residuals_fit_price'])
                testing_labels = labels(ft['test']['residuals_transform_price'])
                ft['train']['labels'] = training_labels
                ft['test']['labels'] = testing_labels

            # Finally, perform our feature scaling
            if weight:
                scaler = sklearn.preprocessing.MinMaxScaler()
                x_scale = scaler.transformFit(ft['train']['ds'])
                y_scale = scaler.transform(ft['test']['ds'])
                ds_scale_x = pd.DataFrame(x_scale)
                ds_scale_y = pd.DataFrame(y_scale)
                ft['train']['ds_scale'] = ds_scale_x
                ft['test']['ds_scale'] = ds_scale_y

            final_list.append(ft)
        self.fts = final_list
        return final_list
    