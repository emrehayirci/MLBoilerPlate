# data analysis and wrangling
import pandas as pd
from pylab import *
# machine learning
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from modelSelection import test
from preprocessing.encoding import encode_features
# Kickstart
from preprocessing.simplification import transform_features
from preprocessing.featureSelection import RecursiveFeatureSelection
# visualization
# from IPython import get_ipython
# get_ipython().run_line_magic('matplotlib', 'qt')

train_df = pd.read_csv('./input/train.csv')
test_df = pd.read_csv('./input/test.csv')


#Get Insights about data
#getDataInsight(train_df)
#analyzeFeaturesByPivoting(train_df, "Survived")

#simplify data
train_df = transform_features(train_df)
test_df = transform_features(test_df)

#analyzeFeaturesByPivoting(train_df, "Survived")

#Convert categoricals to numerical
train_df, test_df = encode_features(train_df, test_df)

#drop Id and create df for testing
Y_train = train_df['Survived']
X_train = train_df.drop(['Survived', 'PassengerId'], axis=1)

X_test = test_df.drop('PassengerId', axis=1).copy()

#Model Selection
test(X_train, Y_train)
#FEATURE SELECTION
""""
#Train
logreg = LogisticRegression()
logregrfe = RecursiveFeatureSelection(logreg, 3)
acc_log = round(svc_model.score(X_train, Y_train) * 100, 2)
#Y_pred = logregrfe.predict(X_test)
Y_pred = svc_model.predict(X_test)
"""

#Best Selected Model Evaluation
forest = RandomForestClassifier(bootstrap=True, max_depth=7, max_features='sqrt', min_samples_leaf=3,
                                min_samples_split=2, n_estimators=10)
forest.fit(X_train, Y_train)
acc_log = forest.score(X_train, Y_train)
Y_pred = forest.predict(X_test)
print(acc_log)

#FINALIZATION
submission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": Y_pred
    })
submission.to_csv('output/'+ "lol-name" + '.csv', index=False)