import numpy as np
import warnings
from readers import read_dataset
from classification import *
from imbalanceLearning import all_imblearn

with warnings.catch_warnings():
    
    X, Y = read_dataset()
    resampledlist = all_imblearn(X, Y)
    warnings.simplefilter("ignore")
    for i in range(len(resampledlist)):
        xx, yy, imblearn = resampledlist[i]
        results, table = classic_classifiers(xx, yy, imblearn)
        print(table)


