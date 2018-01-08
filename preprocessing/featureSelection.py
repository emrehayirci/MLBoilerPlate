from sklearn.feature_selection import RFE, RFECV
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

def RecursiveFeatureSelectionCrossValidated(model):
    rfe = RFECV(model, cv=5, step=1)
    return rfe

def RecursiveFeatureSelection(model):
    rfe = RFE(model, step=1)
    return rfe

def UnivariateFeatureSelection(K, configuration=chi2):
    model = SelectKBest(score_func=configuration, k=K)
    return model