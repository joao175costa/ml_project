import datetime
import os
import pickle

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from readers import read_dataset

os.chdir('..')  # allows saving on the results folder

def optimize_classic_classifiers(screening=''):
    """
    Tunes classic classifiers and outputs average cross-validation results for comparasion.
    Classifers used are: NaiveBayes, RandomForest, LogisticRegression, LDA, KNN and SVC
    Each classifier is hypertuned for each split.
    Classifiers are trained for 10 random train/test splits.
    :param screening: what screening features to add to the feature space, e.g., '' means no screening information is
        used for training
    :return: average metrics for each classifier
    """
    n_shuffles = 10
    X, Y = read_dataset(screening=screening)
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

    now = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M')
    with open('./results/classic_results_' + now + '_' +screening + '.pkl', 'wb') as f:
        pickle.dump(results, f, -1)
    table_of_results.to_csv('./results/classic_results_' + now + '_' + screening +'.csv')

    return results, table_of_results


optimize_classic_classifiers()
