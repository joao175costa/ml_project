import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, minmax_scale
import os

dataset_file = os.curdir + '/dataset/risk_factors_cervical_cancer.csv'


def read_dataset(handle_sparse=True, preprocess='uniform', screening='HSC'):
    """
    reads the risk factors cervical cancer dataset utilizing numpy.
    handle_sparse=True substitutes the absent features for their dataset average
    Biopsy results are considered as target variable
    :returns: data (n_samples*n_feats), labels (n_samples)
    """

    raw_data = np.genfromtxt(dataset_file, delimiter=',', skip_header=1, missing_values='?')
    data = raw_data[:, :32] # remove screening and biopsies
    labels = raw_data[:, -1]

    for method in screening:
        if method == 'H':
            data = np.hstack((data, raw_data[:, 32][:, None]))
        elif method == 'S':
            data = np.hstack((data, raw_data[:, 33][:, None]))
        elif method == 'C':
            data = np.hstack((data, raw_data[:, 34][:, None]))

    if handle_sparse:
        means = np.nanmean(data, axis=0)  # means of columns, ignoring nan
        ix = np.where(np.isnan(data))  # indices of NaNs
        data[ix] = np.take(means, ix[1])  # replace NaN with mean

    if preprocess == 'uniform':
        data = minmax_scale(data)

    return data, labels
