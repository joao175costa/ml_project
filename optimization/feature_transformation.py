import pickle

import numpy as np
from imblearn.combine import SMOTEENN
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap, LocallyLinearEmbedding, MDS
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from readers import read_dataset


def optimize_dimensionality_reduction():
    """
    Optimizes the dimensionality reduction block, namely reduction/transformation methods and its parameters.
    The methods used are Isomap, MDS, LocalllyLinearEmbeddings (transformation), and PCA (reduction)

    This optimization step is performed after applying value imputation, SMOTEENN on the training set,
    and feature selection according to the best found features on SBFS.

    The resulting GridSearchCV with results for each option is stored in a .pkl file
    """

    sbfs = pickle.load(open('smoteenn_sffs.pkl', 'rb'))

    scoring = ['accuracy',
               'precision',
               'recall',
               'balanced_accuracy',
               'average_precision',
               'brier_score_loss',
               'neg_log_loss']
    (X, Y), feature_names = read_dataset(screening='')
    X, Y = SMOTEENN().fit_resample(X, Y)
    X = sbfs.transform(X)

    clf = sbfs.estimator
    feature_mask = list(sbfs.k_feature_idx_)

    pipe = Pipeline([('transform', None),
                     ('classifier', clf)])

    params = [{},

              {'transform': [Isomap(n_jobs=-1),
                             PCA(),
                             MDS(),
                             LocallyLinearEmbedding(n_jobs=-1)],
              'transform__n_components': np.arange(2, len(feature_mask), 2)}
              ]

    grid = GridSearchCV(estimator=pipe,
                        param_grid=params,
                        scoring=scoring,
                        n_jobs=-1,
                        refit='balanced_accuracy',
                        cv=5,
                        verbose=10,
                        error_score=0)
    grid.fit(X, Y)

    print(grid.best_params_)
    pickle.dump(grid, open('transform.pkl', 'wb'), -1)


optimize_dimensionality_reduction()
