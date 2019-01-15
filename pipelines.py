import pickle

import numpy as np
from imblearn.combine import SMOTEENN
from sklearn import metrics
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit, StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from readers import read_dataset, read_screening_dataset

sffs_path = './optimization/smoteenn_sffs.pkl'


def cervical_cancer_classification(screening=''):
    """
    Performs classification on the cervical cancer risk dataset
    :param screening:
    :return:
    """
    sbfs = pickle.load(open(sffs_path, 'rb'))

    (X, Y), feature_names = read_dataset(handle_sparse='mean', screening='HSC')
    clf = SVC(C=1000, kernel='rbf', probability=True, gamma='scale')
    feature_mask_sfs = list(sbfs.k_feature_idx_)
    for s in screening:
        if s == 'H':
            feature_mask_sfs.append(32)
        elif s == 'S':
            feature_mask_sfs.append(33)
        elif s == 'C':
            feature_mask_sfs.append(34)

    sss = StratifiedShuffleSplit(n_splits=10, test_size=0.3)

    results = []
    for train_ix, test_ix in sss.split(X, Y):
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
        Ypred = clf.predict(Xtest)
        Yprob = clf.predict_proba(Xtest)

        # metrics calculation
        metrics_vector = np.array(
            [metrics.accuracy_score(Ytest, Ypred),
             metrics.balanced_accuracy_score(Ytest, Ypred),
             metrics.precision_score(Ytest, Ypred),
             metrics.recall_score(Ytest, Ypred),
             metrics.average_precision_score(Ytest, Yprob[:, 1]),
             metrics.log_loss(Ytest, Yprob[:, 1]),
             ])
        results.append(metrics_vector)
        #print(metrics_vector)

    metric_names = ['Accuracy', 'Balanced Accuracy', 'Precision', 'Recall', 'Average Precision', 'Log Loss']
    results = np.array(results)
    average_results = np.average(results, axis=0)
    average_results = {m: a for (m, a) in zip(metric_names, average_results)}
    return average_results


def risk_screening_classification():
    """
    Performs testing of pipeline for risk + screening classification.
    The original dataset is divided in half (stratified), using one part for training to obtain risk-factor
    probabilities (with SMOTEENN, feature selection)
    Probabilities for the other half are determined using the trained and tuned classifier (SVC).
    Probabilities are appended to the 3 screening features, originating a 4-feature space.
    This new set is then divided in half (stratified) for a new classifier training and testing (GaussianNB).
    Thus: 50% of data is used to train the risk probability classifier, and 50% of that data is used to train the
    risk + screening classifier. The final tested samples were never used for the training of either classifier,
    providing an unbiased result.

    This process is repeated 10 times, with randomly determined partitions.
    The metrics for each fold are recorded and pickled.
    """
    (X, Y), feature_names = read_dataset(handle_sparse='mice', screening='HSC')
    Xscreening = X[:, 32:]
    X = X[:, :32]
    n_repeats = 10

    feature_mask_sfs = list(pickle.load(open(sffs_path, 'rb')).k_feature_idx_)
    clf = SVC(C=1000, kernel='rbf', gamma='scale', probability=True)
    repeated_shuffle = StratifiedShuffleSplit(n_splits=n_repeats, test_size=.5)

    results = []
    for repeat_n, (train_ix, test_ix) in enumerate(repeated_shuffle.split(X, Y)):
        # print('repeat', repeat_n)

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
        # prediction for remaining 50% of dataset, not used for training
        Yprob = clf.predict_proba(Xtest)

        shuffle_split = StratifiedShuffleSplit(n_splits=n_repeats, test_size=.5)
        Xrisk = np.hstack((Yprob, Xscreening[test_ix]))  # 4-feature space
        Yrisk = Ytest
        for fold_n, (train_risk_ix, test_risk_ix) in enumerate(shuffle_split.split(Xtest, Ytest)):
            # print('fold', fold_n)
            # divide dataset not used for probability training in 2:
            # 50% to train risk+screening classifier
            # 50% to test said classifier (25% of full dataset, never used for training)
            Xtrain_risk = Xrisk[train_risk_ix]
            Ytrain_risk = Yrisk[train_risk_ix]
            Xtest_risk = Xrisk[test_risk_ix]
            Ytest_risk = Yrisk[test_risk_ix]

            clf_risk = GaussianNB()
            clf_risk.fit(Xtrain_risk, Ytrain_risk)
            Ypred = clf_risk.predict(Xtest_risk)
            Yprob = clf_risk.predict_proba(Xtest_risk)

            metrics_vector = np.array(
                [metrics.accuracy_score(Ytest_risk, Ypred),
                 metrics.balanced_accuracy_score(Ytest_risk, Ypred),
                 metrics.precision_score(Ytest_risk, Ypred),
                 metrics.recall_score(Ytest_risk, Ypred),
                 metrics.average_precision_score(Ytest_risk, Yprob[:, 1]),
                 metrics.log_loss(Ytest_risk, Yprob[:, 1]),
                 ])
            results.append(metrics_vector)

    pickle.dump(results, open('results/risk_classification_kfold.pkl', 'wb'), -1)

    metric_names = ['Accuracy', 'Balanced Accuracy', 'Precision', 'Recall', 'Average Precision', 'Log Loss']
    results = np.array(results)
    average_results = np.average(results, axis=0)
    average_results = {m: a for (m, a) in zip(metric_names, average_results)}
    return average_results


def risk_screening_dataset_classification():
    """
    Classifies the dataset produced by create_risk_probability_dataset.py, with the selection of the best parameters and
    classifiers from: NaiveBayes, DecisionTree, RandomForest, LogisticRegression, LDA, KNN and SVC

    Saves the cross-validation results to /results folder.

    Best classifier for this task is usually NaiveBayes.
    """
    X, Y = read_screening_dataset()

    pipe = Pipeline([('clf', None)])
    pipe_params = [{'clf': [GaussianNB()]},

                   {'clf': [DecisionTreeClassifier()]},

                   {'clf': [RandomForestClassifier(n_estimators=100)]},

                   {'clf': [LogisticRegression(solver='lbfgs')],
                    'clf__C': np.logspace(-3, 3, 7)},

                   {'clf': [LinearDiscriminantAnalysis(solver='lsqr')],
                    'clf__n_components': [1, 2, 3]},

                   {'clf': [KNeighborsClassifier()],
                    'clf__n_neighbors': [3, 5, 7, 11]},

                   {'clf': [SVC(gamma='scale', probability=True)],
                    'clf__C': np.logspace(-3, 3, 7),
                    'clf__kernel': ['linear', 'rbf']}]

    scoring = ['accuracy',
               'precision',
               'recall',
               'balanced_accuracy',
               'average_precision',
               'brier_score_loss',
               'neg_log_loss']

    cv = 10  # internal number of folds for cross validation

    grid = GridSearchCV(estimator=pipe,
                        param_grid=pipe_params,
                        scoring=scoring,
                        refit='balanced_accuracy',
                        cv=cv,
                        n_jobs=-1,
                        verbose=10)

    grid.fit(X, Y)

    print('best params', grid.best_params_)
    print('best scores', grid.best_score_)

    pickle.dump(grid, open('results/risk_screening_classification.pkl', 'wb'), -1)

