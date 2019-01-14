import numpy as np
from sklearn.preprocessing import minmax_scale
from fancyimpute import IterativeImputer
import os

dataset_file = os.curdir + '/dataset/risk_factors_cervical_cancer.csv'  # path for cervical cancer risk dataset
screening_dataset_file = os.curdir + '/dataset/risk_screening_dataset.csv'  # path for created risk probability dataset


def read_dataset(handle_sparse='mean', preprocess='uniform', screening='HSC'):
    """
    reads the risk factors cervical cancer dataset utilizing numpy.
    Args:
        handle_sparse:
            determines method for dealing with missing data.
            'mean' replaces missing feature values by their dataset mean
            'mice' performs IterativeImputer to extrapolate missing values
        preprocess:
            None, without scaling of features
            'uniform', features are scaled to [0, 1] interval
        screening:
            choose which screening methods to use: Hinselmann (H), Schiller(S) and Citology(C)
            i.e., screening='' returns a dataset without screening features
            and screening='HSC' returns the full dataset

    :returns:
        (data (n_samples*n_feats), labels (n_samples)),
        feature_names (n_feats)
    """

    raw_data = np.genfromtxt(dataset_file, delimiter=',', missing_values='?', names=True)
    raw_feature_names = raw_data.dtype.names
    raw_data = np.genfromtxt(dataset_file, delimiter=',', missing_values='?', skip_header=1)

    data = raw_data[:, :32]  # remove screening and biopsies
    feature_names = list(raw_feature_names[:32])

    labels = raw_data[:, -1]

    if handle_sparse == 'mean':
        means = np.nanmean(data, axis=0)  # means of columns, ignoring nan
        ix = np.where(np.isnan(data))  # indices of NaNs
        data[ix] = np.take(means, ix[1])  # replace NaN with mean

    elif handle_sparse == 'mice':
        n_imputations = 5
        XY_completed = []

        for i in range(n_imputations):
            imputer = IterativeImputer(n_iter=5, sample_posterior=True, random_state=i)
            XY_completed.append(imputer.fit_transform(raw_data))

        XY_completed_mean = np.mean(XY_completed, 0)

        data = XY_completed_mean[:, :32]
        labels = XY_completed_mean[:, -1]

    for method in screening:
        # go through screening methods and adds them to the feature matrix
        if method == 'H':
            data = np.hstack((data, raw_data[:, 32][:, None]))
            feature_names.append(raw_feature_names[32])
        elif method == 'S':
            data = np.hstack((data, raw_data[:, 33][:, None]))
            feature_names.append(raw_feature_names[33])
        elif method == 'C':
            data = np.hstack((data, raw_data[:, 34][:, None]))
            feature_names.append(raw_feature_names[34])

    if preprocess == 'uniform':
        data = minmax_scale(data)

    return (data, labels), feature_names


def read_screening_dataset():
    """
    reads the created risk probability + screening dataset using numpy
    :return:
        data, X matrix, 4 features (risk probability, hinselmann, schiller, citology)
        labes, Y matrix, biopsy
    """

    raw_data = np.genfromtxt(screening_dataset_file, delimiter=',', skip_header=1)
    data = raw_data[:, :-1]
    labels = raw_data[:, -1]

    return (data, labels)