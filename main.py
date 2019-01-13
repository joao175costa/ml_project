import numpy as np
import pickle
import datetime
import warnings
from classification import *

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    # for comb in ['', 'H', 'S', 'C', 'HS', 'HC', 'SC', 'HSC']:
    #     results, table = classic_classifiers(screening=comb)
    #     print(table)
    # feature_selection()

    # select_reduce_classify()
    #smoteenn_sffs_reduction_classify_full()
    #feature_transform()
    #feature_selection_filter()

    risk_probability_and_screening_dataset()
    #final_pipeline()

