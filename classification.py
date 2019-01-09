import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import pickle

from readers import read_dataset

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedShuffleSplit
from sklearn.pipeline import Pipeline, make_pipeline

from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC

from sklearn.manifold import TSNE, Isomap, LocallyLinearEmbedding, SpectralEmbedding, MDS

from sklearn import metrics


def classic_classifiers(save=True, folder_path='results/'):
    n_shuffles = 10
    X, Y = read_dataset()
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

    table = process_results(results)

    if save:
        now = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M')
        with open(folder_path+'classic_results_' + now + '.pkl', 'wb') as f:
            pickle.dump(results, f, -1)
        table.to_csv(folder_path+'classic_results_' + now + '.csv')

    return results, table


def process_results(results):
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
    return table_of_results


def tsne():
    X, Y = read_dataset()
    X_embedded = TSNE().fit_transform(X)
    plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=Y)
    plt.show()
    return 0


def isomap():
    X, Y = read_dataset()
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, stratify=Y, test_size=.5)

    pipe = Pipeline([('embedding', Isomap()), ('clf', SVC(probability=True, gamma='scale'))])
    params = {'embedding__n_components':[2, 5, 10, 20],
              'clf__C': np.logspace(-4, 4, 9)}
    grid_search = GridSearchCV(pipe, cv=5, param_grid=params, n_jobs=-1, verbose=10, refit=True, scoring='recall')
    grid_search.fit(Xtrain, Ytrain)