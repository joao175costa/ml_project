import numpy as np
from sklearn.preprocessing import StandardScaler, minmax_scale
import os

dataset_file = os.curdir + '/dataset/risk_factors_cervical_cancer.csv'


def read_dataset(handle_sparse=True, preprocess='uniform'):
    """
    reads the risk factors cervical cancer dataset utilizing numpy.
    handle_sparse=True substitutes the absent features for their dataset average
    Biopsy results are considered as target variable
    :returns: data (n_samples*n_feats), labels (n_samples)
    """

    raw_data = np.genfromtxt(dataset_file, delimiter=',', skip_header=1, missing_values='?')

    data = raw_data[:, :-1]
    if handle_sparse:
        means = np.nanmean(data, axis=0)  # means of columns, ignoring nan
        ix = np.where(np.isnan(data))  # indices of NaNs
        data[ix] = np.take(means, ix[1])  # replace NaN with mean

    labels = raw_data[:, -1]  # labels generated from the biopsy column

    if preprocess == 'uniform':
        data = minmax_scale(data)

    return data, labels
