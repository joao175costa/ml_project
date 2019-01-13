import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, minmax_scale
import os
from fancyimpute import IterativeImputer

dataset_file = os.curdir + '/dataset/risk_factors_cervical_cancer.csv'


def read_dataset(missing_data='handle_sparse', preprocess='uniform', screening='HSC', mice=False):
    """
    reads the risk factors cervical cancer dataset utilizing numpy.
    handle_sparse=True substitutes the absent features for their dataset average
    Biopsy results are considered as target variable
    :returns: data (n_samples*n_feats), labels (n_samples)
    """

    raw_data = np.genfromtxt(dataset_file, delimiter=',', skip_header=1, missing_values='?')
    data = raw_data[:, :32] # remove screening and biopsies
    labels = raw_data[:, -1]

#    for method in screening:
#        if method == 'H':
#            data = np.hstack((data, raw_data[:, 32][:, None]))
#        elif method == 'S':
#            data = np.hstack((data, raw_data[:, 33][:, None]))
#        elif method == 'C':
#            data = np.hstack((data, raw_data[:, 34][:, None]))
    
    if missing_data == 'handle_sparse':
        means = np.nanmean(data, axis=0)  # means of columns, ignoring nan
        ix = np.where(np.isnan(data))  # indices of NaNs
        data[ix] = np.take(means, ix[1])  # replace NaN with mean
   
    elif missing_data == 'mice':
        n_imputations = 5
        XY_completed = []
        
        for i in range(n_imputations):
            imputer = IterativeImputer(n_iter=5, sample_posterior=True, random_state=i)
            XY_completed.append(imputer.fit_transform(raw_data))
       
        XY_completed_mean = np.mean(XY_completed, 0)
        
        data = XY_completed_mean[:, :32] 
        labels = XY_completed_mean[:, -1]
        
        
    if preprocess == 'uniform':
        data = minmax_scale(data)
        

    return data, labels
