from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, completeness_score

import numpy as np
#General Configuration
scores = ['precision'] # ['precision', 'recall']
cross_validation = StratifiedKFold(n_splits=5)

#SVC Model Configuration
tuned_svc_parameters = [{'kernel': ['rbf'], 'gamma': [1e-4, 1e-5],
                         'C': [100, 1000, 10000]},
                        {'kernel': ['linear'], 'C': [1, 10, 100]}]

#Random Forest Model Configuration
tuned_random_forest_parameters = {
        'max_depth': [5, 6, 7],
        'n_estimators': [2, 5, 10],
        'max_features': ['sqrt', 'auto'], #'log2'],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 3, 5],
        'bootstrap': [True], #, False],
}
#Logistic Regression Model Configuration
tuned_logistic_regression_parameters = {}

def compute_score(clf, X, y, scoring='accuracy', defaul_cv = 5):
    ''' This function calculates the mean accuracy of the prediction of the classifier clf
    on the training set X and against the target values Y using a k fold cross validation.
    It returns the mean accuracy and std
    '''
    xval = cross_val_score(clf, X, y, cv=defaul_cv, scoring=scoring)
    return np.mean(xval), np.std(xval)

def svc_classify(train, targets, score):
    x_train, x_test, y_train, y_test = train_test_split(
    train, targets, test_size=0.4, random_state=0)
    clf = GridSearchCV(SVC(), tuned_svc_parameters, cv=5,
                       scoring='%s_macro' % score)
    clf.fit(x_train, y_train)#

    print("Best parameters set found on development set:")
    print(clf.best_params_)
    print("\nGrid scores on development set:")
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))

    print("\nDetailed classification report:")
    print("\nThe model is trained on the full development set.")
    print("\nThe scores are computed on the full evaluation set.")
    y_true, y_pred = y_test, clf.predict(x_test)
    print(classification_report(y_true, y_pred))


def random_forest_classify(train, targets):
    forest = RandomForestClassifier()
    grid_search = GridSearchCV(forest,
                               scoring='accuracy',
                               param_grid=tuned_random_forest_parameters,
                               cv=cross_validation)

    grid_search.fit(train, targets)
    model = grid_search

    print('Best score: {}'.format(grid_search.best_score_))
    print('Best parameters: {}'.format(grid_search.best_params_))

    mean, std = compute_score(model, train, targets, scoring='accuracy')
    print('Accuracy= %.2f +- %.2f' % (mean, 2 * std))

def multi
def test(train, targets):
    for score in scores:
        print("# Tuning hyper-parameters for %s\n" % score)
        print("Testing Random Forest")
        random_forest_classify(train, targets)
        print("Testing SVC")
        svc_classify(train,targets, score)


