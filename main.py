import numpy as np
import pickle
import datetime
import warnings
from classification import *

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    results, table = classic_classifiers()

print(table)

