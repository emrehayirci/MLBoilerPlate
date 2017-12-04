from pylab import *
from utils import printDefault

# data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd

# visualization
import seaborn as sns
import matplotlib.pyplot as plt

#from IPython import get_ipython
#get_ipython().run_line_magic('matplotlib', 'qt')

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier

#Kickstart

from utils import printDefault
from learning import getDataInsight, analyzeFeaturesByPivoting
from simplification import transform_features
from encoding import encode_features
from featureSelection import RecursiveFeatureSelection, UnivariateFeatureSelection

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

#FEATURE SELECTION

#Train
logreg = LogisticRegression()
logregrfe = RecursiveFeatureSelection(logreg, 2)
hop = logregrfe.fit_transform(X_train, Y_train)
print(X_train.columns)
print(logregrfe.get_support())
Y_pred = logregrfe.predict(X_test)
acc_log = round(logregrfe.score(X_train, Y_train) * 100, 2)

print(acc_log)

submission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": Y_pred
    })
submission.to_csv('output/submission.csv', index=False)