import pickle

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from readers import read_screening_dataset

n_shuffles = 10
X, Y = read_screening_dataset()

ssf = StratifiedShuffleSplit(n_splits=n_shuffles, test_size=.3)  # 10 splits to calculate the average metrics
cv = 5  # internal number of folds for cross validation

pipe = Pipeline([('clf', None)])
pipe_params=[{'clf': [GaussianNB()]},

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