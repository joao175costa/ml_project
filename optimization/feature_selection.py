import pickle
from collections import Counter

import numpy as np
import pandas as pd
from imblearn.combine import SMOTEENN
from mlxtend.feature_selection import SequentialFeatureSelector
from sklearn.feature_selection import mutual_info_classif, SelectKBest, f_classif, chi2, SelectPercentile
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from readers import read_dataset


def optimize_sequential_method():
    """
    Optimizes the feature selection step using a Sequential Backwards Floating Feature Selection method
    The classifier used for this sequential method is SVC, tuned with the best parameters for the whole dataset

    The feature selector is stored in a .pkl file and the results of SFS are stored in a csv format
    """
    (X, Y), feature_names = read_dataset(screening='')  # no screening results, only risk factors

    # dataset resampling for imbalanced data compensation
    smoteenn = SMOTEENN()

    Xres, Yres = smoteenn.fit_resample(X, Y)  # resampled dataset
    print('Resampling')
    print('Original dataset size:', Counter(Y))
    print('Resampled dataset size:', Counter(Yres))

    # feature selection using sequential forward floating selection and tuned SVM
    scoring = ['accuracy',
               'precision',
               'recall',
               'balanced_accuracy',
               'average_precision',
               'brier_score_loss',
               'neg_log_loss']

    param_grid = {'C': np.logspace(-3, 3, 7),
                  'kernel': ['rbf']}

    grid = GridSearchCV(estimator=SVC(probability=True, gamma='scale'),
                       param_grid=param_grid,
                       n_jobs=-1,
                       verbose=10,
                       cv=5,
                       scoring=scoring,
                       refit='balanced_accuracy',
                       iid=False,
                       error_score=0)

    grid.fit(Xres, Yres)
    print(grid.best_params_)

    selector = SequentialFeatureSelector(forward=False,
                                         floating=True,
                                         k_features='best',
                                         verbose=2,
                                         n_jobs=-1,
                                         scoring='balanced_accuracy',
                                         cv=5,
                                         estimator=SVC(probability=True, gamma='scale',
                                                       kernel=grid.best_params_['kernel'],
                                                       C=grid.best_params_['C'])
                                         )

    selector.fit(Xres, Yres, custom_feature_names=feature_names)

    with open('smoteenn_sbfs.pkl', 'wb') as f:
        pickle.dump(selector, f, -1)

    df = pd.DataFrame(selector.subsets_)
    df.to_csv('smoteenn_sbfs.csv')


optimize_sequential_method()


def optimize_filter_method():
    """
    Optimizes the feature selection method using filter methods.
    The classifier used for validation is SVC, tuned with the best classifiers for the whole dataset
    Filter methods are used to assess the number of features to keep, according to scoring functions
    (mutual info, F and Chi2)

    The resulting GridSearchCV is stored in a .pkl file. The best parameters for feature selection are stored within.
    """

    (X, Y), feature_names = read_dataset(screening='')
    Xres, Yres = SMOTEENN().fit_resample(X, Y)

    clf = SVC(C=1000, kernel='rbf', gamma='scale', probability=True)
    scoring = ['accuracy',
               'precision',
               'recall',
               'balanced_accuracy',
               'average_precision',
               'brier_score_loss',
               'neg_log_loss']
    pipe = Pipeline([('filter', None),
                     ('clf', clf)])
    params = [{},
              {'filter': [SelectKBest()],
               'filter__score_func': [mutual_info_classif, f_classif, chi2],
               'filter__k': np.arange(2, 33, 2)}
             ]

    grid = GridSearchCV(estimator=pipe,
                        param_grid=params,
                        scoring=scoring,
                        n_jobs=-1,
                        refit='balanced_accuracy',
                        cv=5,
                        verbose=10,
                        error_score=0)
    grid.fit(Xres, Yres)
    print(grid.best_params_)
    pickle.dump(grid, open('feature_selection_filter.pkl', 'wb'), -1)


optimize_filter_method()
