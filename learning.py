from utils import printDefault
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


CATEGORICAL_FEATURE_MULTIPLIER = 2
"""Used for determining categorical or not.
 How many variable results should be in a column 
 ForExample for 300 len df if we set this to 2 
 maximum categorical variable variety of a column 
 should be 150. If 151 unique variables it is not 
 categorical"""

def getDataInsight(train_df):
    # Get to know your data
    printDefault("Colums", train_df.columns.values)
    printDefault("Sampe of Data", train_df.sample(3))
    printDefault("Head of Data", train_df.head())
    printDefault("Tail of Data", train_df.tail())
    printDefault("General Insight of Data", train_df.info())
    printDefault("Lot more details about data", train_df.describe())
    return

def analyzeFeaturesByPivoting(train_df, targetFeature):
    for column in train_df.columns:

        if (len(train_df[column].unique()) < (len(train_df) / CATEGORICAL_FEATURE_MULTIPLIER)) and (targetFeature != column):
            plt.figure(column)
            sns.countplot(x=column, hue=targetFeature, data=train_df)

# TODO: Add HUE corresponding relationships with all features each other
# TODO: Make Continuous value visualization different