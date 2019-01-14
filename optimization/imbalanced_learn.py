import datetime
import os
import pickle
import warnings

import numpy as np
import pandas as pd
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.over_sampling import RandomOverSampler, SMOTE, SMOTENC
from imblearn.under_sampling import AllKNN, CondensedNearestNeighbour, OneSidedSelection, NeighbourhoodCleaningRule
from imblearn.under_sampling import ClusterCentroids, RandomUnderSampler, NearMiss
from imblearn.under_sampling import EditedNearestNeighbours, RepeatedEditedNearestNeighbours
from sklearn import metrics
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from readers import read_dataset

os.chdir('..')  # allows saving onto the results folder


def inbalanced_classic_classifiers(X, Y, imbalancelearn, save=True):
    """
    Performs hyperparameter tuning on classic classifiers, to determine their best parameters and best imbalanced learn
    strategy.

    :param X: feature matrix, resampled according to imbalancelearn method
    :param Y: label vector, resampled according to imbalancelearn method
    :param imbalancelearn: the method used for imbalance compensation
    :param save: bool, True saves the results in a .pkl file
    :returns: results in a dict and table
    """
    n_shuffles = 10
    ssf = StratifiedShuffleSplit(n_splits=n_shuffles, test_size=.3)  # 10 splits to calculate the average metrics
    cv = 3  # internal number of folds for cross validation

    models = [('Naive_Bayes',
               GaussianNB()),

              ('Forest',
               GridSearchCV(estimator=RandomForestClassifier(),
                            cv=cv,
                            refit=True,
                            n_jobs=-1,
                            param_grid={'n_estimators': [10, 50, 100, 200],
                                        'max_depth': [None, 2, 5, 10]
                                        }
                            )),

              ('LogReg',
               GridSearchCV(estimator=LogisticRegression(max_iter=1000,
                                                         solver='lbfgs'),
                            cv=cv,
                            refit=True,
                            n_jobs=-1,
                            param_grid={'C': np.logspace(-3, 3, num=7)
                                        }
                            )),

              ('LDA',
               GridSearchCV(estimator=LinearDiscriminantAnalysis(solver='lsqr'),
                            cv=cv,
                            refit=True,
                            n_jobs=-1,
                            param_grid={'n_components': [2, 5, 10, 20, 30],
                                        'shrinkage': np.linspace(0, 1, 5)
                                        }
                            )),

              ('KNN',
               GridSearchCV(estimator=KNeighborsClassifier(),
                            cv=cv,
                            refit=True,
                            n_jobs=-1,
                            param_grid={'n_neighbors': [3, 5, 11, 21]
                                        }
                            )),

              ('SVM',
               GridSearchCV(estimator=SVC(gamma='scale',
                                          probability=True
                                          ),
                            cv=cv,
                            refit=True,
                            n_jobs=-1,
                            param_grid={'C': np.logspace(-3, 3, num=7),
                                        'kernel': ['rbf', 'linear'],
                                        }
                            ))
              ]

    results = {m_name: np.zeros(6) for m_name, _ in models}
    results.update({m_name+'_params': list() for m_name, m in models if type(m) == GridSearchCV})

    # perform train_test_split 10 times to achieve an average result
    for fold_n, (train_ix, test_ix) in enumerate(ssf.split(X, Y)):
        print('\nfold number', fold_n)

        Xtrain = X[train_ix]
        Ytrain = Y[train_ix]
        Xtest = X[test_ix]
        Ytest = Y[test_ix]

        for model_name, model in models:
            print(model_name)
            model.fit(Xtrain, Ytrain)
            Ypred = model.predict(Xtest)
            Yprob = model.predict_proba(Xtest)

            pos_ratio = np.mean(Ytest == 1)
            sample_weight = pos_ratio * np.ones(Ytest.shape[0])
            sample_weight[Ytest == 1] = 1. - pos_ratio

            metrics_vector = np.array(
                [metrics.accuracy_score(Ytest, Ypred),
                 metrics.precision_score(Ytest, Ypred),
                 metrics.recall_score(Ytest, Ypred),
                 metrics.average_precision_score(Ytest, Yprob[:, 1]),
                 metrics.brier_score_loss(Ytest, Yprob[:, 1], sample_weight=sample_weight),
                 metrics.log_loss(Ytest, Yprob[:, 1]),
                 ])

            if type(model) == GridSearchCV:
                results[model_name + '_params'].append(model.best_params_)

            results[model_name] += metrics_vector / n_shuffles  # performs average by summing

    print('Classifiers Trained\n')

    table_of_results = dict()
    for model in results:
        if type(results[model]) == list:
            # parameter lists
            continue
        table_of_results[model] = results[model]
    table_of_results = pd.DataFrame.from_dict(table_of_results,
                                              orient='index',
                                              columns=['Accuracy', 'Precision', 'Recall',
                                                       'avPrecision', 'Brier', 'logLoss'])

    if save:
        now = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M')
        with open('./results/classic_results_' + imbalancelearn + '_' + now + '.pkl', 'wb') as f:
            pickle.dump(results, f, -1)
        table_of_results.to_csv('./results/classic_results_' + imbalancelearn + '_' + now + '.csv')

    return results, table_of_results


def all_imblearn(xx, yy):
    """

    :param xx:
    :param yy:
    :return:
    """

    imblearnlist = []

    """OVER SAMPLING"""

    """Random Over Sampler"""
    ros = RandomOverSampler(random_state=0)
    X_resampled, y_resampled = ros.fit_resample(xx, yy)
    randomOverSampler = [X_resampled, y_resampled, 'random over sampler']
    imblearnlist.append(randomOverSampler)

    """SMOTE"""
    X_resampled, y_resampled = SMOTE().fit_resample(xx, yy)
    smote = [X_resampled, y_resampled, 'smote']
    imblearnlist.append(smote)

    """SMOTE borderline1"""
    sm = SMOTE(kind='borderline1')
    X_resampled, y_resampled = sm.fit_resample(xx, yy)
    smote = [X_resampled, y_resampled, 'smote borderline1']
    imblearnlist.append(smote)

    """SMOTE borderline2"""
    sm = SMOTE(kind='borderline2')
    X_resampled, y_resampled = sm.fit_resample(xx, yy)
    smote = [X_resampled, y_resampled, 'smote borderline2']
    imblearnlist.append(smote)

    """SMOTE svm"""
    sm = SMOTE(kind='svm')
    X_resampled, y_resampled = sm.fit_resample(xx, yy)
    smote = [X_resampled, y_resampled, 'smote svm']
    imblearnlist.append(smote)

    """SMOTENC"""
    smote_nc = SMOTENC(categorical_features=[0, 2], random_state=0)
    X_resampled, y_resampled = smote_nc.fit_resample(xx, yy)
    smote = [X_resampled, y_resampled, 'smotenc']
    imblearnlist.append(smote)

    #    """ADASYN"""
    #    X_resampled, y_resampled = ADASYN.fit_resample(xx, yy)
    #    adasyn = [X_resampled, y_resampled, 'adasyn']
    #    imblearnlist.append(adasyn)
    #

    """UNDER SAMPLING"""

    """Cluster Centroids"""
    cc = ClusterCentroids(random_state=0)
    X_resampled, y_resampled = cc.fit_resample(xx, yy)
    reSampled = [X_resampled, y_resampled, 'cluster centroids']
    imblearnlist.append(reSampled)

    """Random Over Sampler"""
    rus = RandomUnderSampler(random_state=0)
    X_resampled, y_resampled = rus.fit_resample(xx, yy)
    reSampled = [X_resampled, y_resampled, 'random under sampler']
    imblearnlist.append(reSampled)

    """Near Miss 1"""
    nm1 = NearMiss(version=1)
    X_resampled, y_resampled = nm1.fit_resample(xx, yy)
    reSampled = [X_resampled, y_resampled, 'near miss 1']
    imblearnlist.append(reSampled)

    """Near Miss 2"""
    nm2 = NearMiss(version=2)
    X_resampled, y_resampled = nm2.fit_resample(xx, yy)
    reSampled = [X_resampled, y_resampled, 'near miss 2']
    imblearnlist.append(reSampled)

    """Near Miss 3"""
    nm3 = NearMiss(version=3)
    X_resampled, y_resampled = nm3.fit_resample(xx, yy)
    reSampled = [X_resampled, y_resampled, 'near miss 3']
    imblearnlist.append(reSampled)

    """Edited Nearest Neighbours"""
    enn = EditedNearestNeighbours()
    X_resampled, y_resampled = enn.fit_resample(xx, yy)
    reSampled = [X_resampled, y_resampled, 'edited nearest neighbours']
    imblearnlist.append(reSampled)

    """Repeated Edited Nearest Neighbours"""
    renn = RepeatedEditedNearestNeighbours()
    X_resampled, y_resampled = renn.fit_resample(xx, yy)
    reSampled = [X_resampled, y_resampled, 'repeated edited nearest neighbours']
    imblearnlist.append(reSampled)

    """All KNN"""
    allknn = AllKNN()
    X_resampled, y_resampled = allknn.fit_resample(xx, yy)
    reSampled = [X_resampled, y_resampled, 'allKNN']
    imblearnlist.append(reSampled)

    """Condensed Nearest Neighbour"""
    cnn = CondensedNearestNeighbour(random_state=0)
    X_resampled, y_resampled = cnn.fit_resample(xx, yy)
    reSampled = [X_resampled, y_resampled, 'Condensed Nearest Neighbour']
    imblearnlist.append(reSampled)

    """One Sided Selection"""
    oss = OneSidedSelection(random_state=0)
    X_resampled, y_resampled = oss.fit_resample(xx, yy)
    reSampled = [X_resampled, y_resampled, 'One Sided Selection']
    imblearnlist.append(reSampled)

    """Neighbourhood Cleaning Rule"""
    ncr = NeighbourhoodCleaningRule()
    X_resampled, y_resampled = ncr.fit_resample(xx, yy)
    reSampled = [X_resampled, y_resampled, 'Neighbourhood Cleaning Rule']
    imblearnlist.append(reSampled)

    """OVER AND UNDER SAMPLING"""

    """SMOTEENN"""
    smote_enn = SMOTEENN(random_state=0)
    X_resampled, y_resampled = smote_enn.fit_resample(xx, yy)
    reSampled = [X_resampled, y_resampled, 'SMOTEENN']
    imblearnlist.append(reSampled)

    """SMOTETomek"""
    smote_tomek = SMOTETomek(random_state=0)
    X_resampled, y_resampled = smote_tomek.fit_resample(xx, yy)
    reSampled = [X_resampled, y_resampled, 'SMOTETomek']
    imblearnlist.append(reSampled)

    return imblearnlist


X, Y = read_dataset()
resampledlist = all_imblearn(X, Y)
warnings.simplefilter("ignore")
for i in range(len(resampledlist)):
    xx, yy, imblearn = resampledlist[i]
    results, table = inbalanced_classic_classifiers(xx, yy, imblearn)
    print(table)
