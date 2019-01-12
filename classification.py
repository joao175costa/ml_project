import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import pickle
from collections import Counter

from imblearn.combine import SMOTEENN
from mlxtend.feature_selection import SequentialFeatureSelector

from readers import read_dataset

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedShuffleSplit, StratifiedKFold
from sklearn.pipeline import Pipeline, make_pipeline

from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC

from sklearn.manifold import TSNE, Isomap, LocallyLinearEmbedding, SpectralEmbedding, MDS

from sklearn.feature_selection import RFECV, SelectPercentile, mutual_info_classif, SelectKBest

from sklearn import metrics


def classic_classifiers(save=True, folder_path='results/', screening='HSC'):
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
    return 0


def isomap():
    X, Y = read_dataset()
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, stratify=Y, test_size=.5)

    pipe = Pipeline([('embedding', Isomap()), ('clf', SVC(probability=True, gamma='scale'))])
    params = {'embedding__n_components':[2, 5, 10, 20],
              'clf__C': np.logspace(-4, 4, 9)}
    grid_search = GridSearchCV(pipe, cv=5, param_grid=params, n_jobs=-1, verbose=10, refit=True, scoring='recall')
    grid_search.fit(Xtrain, Ytrain)


def feature_selection():
    X, Y = read_dataset()
    feature_names = np.array(list(pd.read_csv('./dataset/risk_factors_cervical_cancer.csv', header=0, index_col=None))[:-1])

    selector = SelectPercentile(score_func=mutual_info_classif, percentile=50)
    selector.fit(X, Y)
    scores = selector.scores_
    removed = np.logical_not(selector.get_support())
    print('removed features', feature_names[removed])
    scores, feature_names = zip(*sorted(zip(scores, feature_names)))
    plt.figure()
    plt.barh(feature_names, scores)
    plt.show()


def select_reduce_classify():
    X, Y = read_dataset(screening='')

    # perform train_test_split 10 times to achieve an average result

    pipe = Pipeline([('selection', None),
                     ('reduction', None),
                     ('classification', SVC(gamma='scale', probability=True))])

    pipe_params = [{
                    'classification__C': np.logspace(-3, 3, num=7),
                    'classification__kernel': ['rbf', 'linear']},

                   {
                    'selection': [SelectKBest(score_func=mutual_info_classif)],
                    'selection__k': np.arange(5, 36, 5),
                    'classification__C': np.logspace(-3, 3, num=7),
                    'classification__kernel': ['rbf', 'linear']},

                   {
                    'reduction': [Isomap(n_jobs=-1)],
                    'reduction__n_components': np.arange(5, 36, 5),
                    'classification__C': np.logspace(-3, 3, num=7),
                    'classification__kernel': ['rbf', 'linear']},

                   {
                    'selection': [SelectKBest(score_func=mutual_info_classif)],
                    'selection__k': np.arange(5, 36, 5),
                    'reduction': [Isomap(n_jobs=-1)],
                    'reduction__n_components': np.arange(5, 36, 5),
                    'classification__C': np.logspace(-3, 3, num=7),
                    'classification__kernel': ['rbf', 'linear']}

                   ]

    search = GridSearchCV(estimator=pipe,
                          param_grid=pipe_params,
                          cv=3,
                          refit='average_precision',
                          n_jobs=-1,
                          iid=False,
                          verbose=10,
                          scoring=['accuracy',
                                   'precision',
                                   'recall',
                                   'balanced_accuracy',
                                   'average_precision',
                                   'brier_score_loss',
                                   'neg_log_loss'],
                          error_score=0)

    search.fit(X, Y)

    results = search.cv_results_

    print(search.best_params_)

    now = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M')
    with open('results/pipeline_' + now + '.pkl', 'wb') as f:
        pickle.dump(results, f, -1)


def smoteenn_sffs_reduction_classify_full():

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


    param_grid = {'C': np.logspace(-1, 6, 8),
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

    clf = SVC(probability=True, gamma='scale', **grid.best_params_)

    selector = SequentialFeatureSelector(forward=True,
                                         floating=True,
                                         k_features='best',
                                         verbose=2,
                                         n_jobs=-1,
                                         cv=5,
                                         estimator=clf)

    selector.fit(Xres, Yres, custom_feature_names=feature_names)

    print(selector.subsets_)

    with open('smoteenn_sfs.pkl', 'wb') as f:
        pickle.dump(selector, f, -1)




