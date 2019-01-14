"""
Creates the risk probability + screening dataset from the original dataset and implemented pipeline.
Missing values are completed using IterativeImputator.
Class imbalance is corrected on the training set using SMOTEENN.
Feature selection is performed according to the results of SBFS.
Feature transformation/reduction is not performed, since it has poorer results.
The probability of positive class for each sample is obtained through a Leave-One-Out (LOO) strategy, i.e.,
all dataset samples except the selected one are used for classifier training, from which the probability of the
selected sample is obtained.

The computed probability is appended to the remaining 3 screening features (Hinselmann, Schiller and Citology), thus
creating a 4-feature space. Target label (Biopsy) is also appended for each sample.

The resulting dataset is saved in .csv format on the /dataset folder.
"""
import pickle

import numpy as np
import pandas as pd
from imblearn.combine import SMOTEENN
from sklearn.model_selection import LeaveOneOut
from sklearn.svm import SVC

from readers import read_dataset

# read the risk factors dataset, divide into risk factor features and screening features
(X, Y), feature_names = read_dataset(handle_sparse='mice', screening='HSC')
Xscreening = X[:, 32:]
n_samples = len(Y)
X = X[:, :32]

# feature mask obtained from Selective Backwards Floating Feature Selection
feature_mask_sfs = list(pickle.load(open('smoteenn_sbfs.pkl', 'rb')).k_feature_idx_)

# classifier used, with tuned parameters according to the whole dataset
clf = SVC(C=1000, kernel='rbf', gamma='scale', probability=True)
loo = LeaveOneOut()

# initialization of the final dataset X matrix
Xrisk_screening = np.hstack((np.zeros(n_samples)[:, None], Xscreening, Y[:, None]))

# go through all samples, using remaining ones for training
for train_ix, test_ix in loo.split(X, Y):
    print(test_ix)
    Xtrain = X[train_ix]
    Ytrain = Y[train_ix]
    Xtest = X[test_ix]
    Ytest = Y[test_ix]

    # imbalance compensation
    Xtrain, Ytrain = SMOTEENN().fit_resample(Xtrain, Ytrain)

    # feature selection
    Xtrain = Xtrain[:, feature_mask_sfs]
    Xtest = Xtest[:, feature_mask_sfs]

    # classification with tuned classifier
    clf.fit(Xtrain, Ytrain)
    Yprob = clf.predict_proba(Xtest)  # predict probability with classifier trained with remaining samples
    Xrisk_screening[test_ix, 0] = Yprob[0, 1]

df = pd.DataFrame(Xrisk_screening, index=None, columns=['Risk', 'Hinselmann', 'Schiller', 'Citology', 'Biospy'])
df.to_csv('dataset/risk_screening_dataset.csv', index=False)