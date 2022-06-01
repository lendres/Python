"""
Created on April 9, 2022
@author: Lance A. Endres
"""
# To help with reading and manipulation of data
import numpy                         as np
import pandas                        as pd

# To help with data visualization
import matplotlib.pyplot             as plt
import seaborn                       as sns


from lendres.PlotHelper              import PlotHelper

# To get different performance metrics
import sklearn.metrics               as metrics
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    recall_score,
    accuracy_score,
    precision_score,
    f1_score
)


# defining a function to compute different metrics to check performance of a classification model built using sklearn
def model_performance_classification_sklearn(model, predictors, target):
    """
    Function to compute different metrics to check classification model performance

    model: classifier
    predictors: independent variables
    target: dependent variable
    """

    # predicting using the independent variables
    pred = model.predict(predictors)

    accuracy     = accuracy_score(target, pred)
    recall       = recall_score(target, pred)
    precision    = precision_score(target, pred)
    f1           = f1_score(target, pred)

    # creating a dataframe of metrics
    df_perf = pd.DataFrame(
        {
            "Accuracy"  : accuracy,
            "Recall"    : recall,
            "Precision" : precision,
            "F1"        : f1,
        },
        index=[0],
    )

    return df_perf


def model_performance_classification(trueValues, predictedValues):
    """
    Function to compute different metrics to check classification model performance.  Uses
    the true (specified) and predicted values as input.

    trueValues: independent variable true values
    predictedValues: independent variable predicted values
    """
    accuracy     = accuracy_score(trueValues, predictedValues)
    recall       = recall_score(trueValues, predictedValues)
    precision    = precision_score(trueValues, predictedValues)
    f1           = f1_score(trueValues, predictedValues)

    # creating a dataframe of metrics
    performanceDataFrame = pd.DataFrame(
        {
            "Accuracy"  : accuracy,
            "Recall"    : recall,
            "Precision" : precision,
            "F1"        : f1,
        },
        index=[0],
    )

    return performanceDataFrame


def confusion_matrix_sklearn(model, predictors, target, titlePrefix=None):
    """
    To plot the confusion_matrix with percentages

    model: classifier
    predictors: independent variables
    target: dependent variable
    """
    y_pred = model.predict(predictors)
    cm     = confusion_matrix(target, y_pred)

    labels = np.asarray(
        [
            ["{0:0.0f}".format(item) + "\n{0:.2%}".format(item / cm.flatten().sum())]
            for item in cm.flatten()
        ]
    ).astype("object").reshape(2, 2)

    # Tack on the type labels to the numerical information.
    labels[0, 0] += "\nTN"
    labels[1, 0] += "\nFN\nType 2"
    labels[0, 1] += "\nFP\nType 1"
    labels[1, 1] += "\nTP"

    PlotHelper.FormatPlot(width=5.35, height=4)
    #plt.figure(figsize=(6, 4))
    axis = sns.heatmap(cm, annot=labels, annot_kws={"fontsize" : 14*PlotHelper.scale}, fmt="")
    PlotHelper.Label(axis, title="Data", xLabel="Predicted", yLabel="Actual", titlePrefix=titlePrefix)
    plt.show()