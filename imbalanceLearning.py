import numpy as np
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN, SMOTENC
from imblearn.under_sampling import ClusterCentroids, RandomUnderSampler, NearMiss
from imblearn.under_sampling import EditedNearestNeighbours, RepeatedEditedNearestNeighbours
from imblearn.under_sampling import AllKNN, CondensedNearestNeighbour, OneSidedSelection, NeighbourhoodCleaningRule
from imblearn.under_sampling import InstanceHardnessThreshold
from imblearn.combine import SMOTEENN, SMOTETomek


def all_imblearn(xx, yy):
    
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
    
    
    
    
    
    
    