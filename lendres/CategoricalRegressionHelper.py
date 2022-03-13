# -*- coding: utf-8 -*-
"""
Created on Wed Jan 19 07:49:25 2022

@author: Lance
"""

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import metrics

from lendres.ConsoleHelper import ConsoleHelper
from lendres.ModelHelper import ModelHelper
from lendres.PlotHelper import PlotHelper

class CategoricalRegressionHelper(ModelHelper):

    def __init__(self, dataHelper):
        """
        Constructor.

        Parameters
        ----------
        dataHelper : DataHelper
            DataHelper that has the data in a pandas.DataFrame.

        Returns
        -------
        None.
        """
        super().__init__(dataHelper)


    def CreateFeatureImportancePlot(self, titlePrefix=None):
        """
        Plots importance factors as a bar plot.

        Parameters
        ----------
        titlePrefix : string or None, optional
            If supplied, the string is prepended to the title.

        Returns
        -------
        None.
        """
        # Need the values in the reverse order (smallest to largest) for the bar plot to get the largest value on
        # the top (highest index position).
        importancesDataFrame = self.GetSortedImportance(ascending=True)
        indices              = range(importancesDataFrame.shape[0])

        # Must be run before creating figure or plotting data.
        PlotHelper.FormatPlot()

        plt.barh(indices, importancesDataFrame["Importance"], color="cornflowerblue", align="center")
        plt.yticks(indices, importancesDataFrame.index, fontsize=12*PlotHelper.scale)
        PlotHelper.Label(plt.gca(), title="Feature Importances", xLabel="Relative Importance", titlePrefix=titlePrefix)

        plt.show()


    def GetSortedImportance(self, ascending=False):
        """
        Sorts the importance factors and returns them in a Pandas DataFrame.

        Parameters
        ----------
        ascending : bool
            Specifies if the values should be sorted as ascending or descending.

        Returns
        -------
        : pandas.DataFrame
            DataFrame of the sorted importance values.
        """
        return pd.DataFrame(self.model.feature_importances_,
                            columns=["Importance"],
                            index=self.xTrainingData.columns).sort_values(by="Importance", ascending=ascending)


    def CreateConfusionMatrixPlot(self, dataSet="training", titlePrefix=None):
        """
        Plots the confusion matrix for the model output.

        Parameters
        ----------
        dataSet : string
            Which data set(s) to plot.
            training - Plots the results from the training data.
            testing  - Plots the results from the test data.
        titlePrefix : string or None, optional
            If supplied, the string is prepended to the title.

        Returns
        -------
        confusionMatrix : ndarray of shape (n_classes, n_classes)
            The confusion matrix for the data.
        """
        confusionMatrix = self.GetConfusionMatrix(dataSet)

        # The numpy array has to be set as an object type.  If set (or allowed to assume) a
        # type of "str" the entry is created only large enough for the initial string (a character
        # type is used).  It is not possible to append to it.
        labels = np.asarray(
            [
                ["{0:0.0f}".format(item) + "\n{0:.2%}".format(item / confusionMatrix.flatten().sum())]
                for item in confusionMatrix.flatten()
            ]
        ).astype("object").reshape(2, 2)

        # Tack on the type labels to the numerical information.
        labels[0, 0] += "\nTN"
        labels[1, 0] += "\nFN\nType 2"
        labels[0, 1] += "\nFP\nType 1"
        labels[1, 1] += "\nTP"

        # Must be run before creating figure or plotting data.
        # The standard scale for this plot will be a little higher than the normal scale.
        # Not much is shown, so we can shrink the figure size.
        PlotHelper.FormatPlot(width=5.35, height=4)

        # Create plot and set the titles.
        axis = sns.heatmap(confusionMatrix, annot=labels, annot_kws={"fontsize" : 14*PlotHelper.scale}, fmt="")
        PlotHelper.Label(axis, title=dataSet.title()+" Data", xLabel="Predicted", yLabel="Actual", titlePrefix=titlePrefix)

        plt.show()

        return confusionMatrix


    def GetConfusionMatrix(self, dataSet="training"):
        """
        Gets the confusion matrix for the model output.

        Parameters
        ----------
        dataSet : string
            Which data set(s) to plot.
            training - Plots the results from the training data.
            testing  - Plots the results from the test data.
        scale : double
            Scaling parameter used to adjust the plot fonts, lineweights, et cetera for the output scale of the plot.

        Returns
        -------
        confusionMatrix : ndarray of shape (n_classes, n_classes)
            The confusion matrix for the data.
        """
        # Initialize to nothing.
        confusionMatrix = None

        # Get the confusion matrix for the correct data set.
        if dataSet == "training":
            if len(self.yTrainingPredicted) == 0:
                self.Predict()
            confusionMatrix = metrics.confusion_matrix(self.yTrainingData, self.yTrainingPredicted)

        elif dataSet == "testing":
            if len(self.yTestingPredicted) == 0:
                self.Predict()
            confusionMatrix = metrics.confusion_matrix(self.yTestingData, self.yTestingPredicted)

        else:
            raise Exception("Invalid data set specified.")

        return confusionMatrix


    def DisplayModelPerformanceScores(self):
        """
        Displays the model performance scores based on the settings in the ConsuleHelper.

        Returns
        -------
        None.
        """
        scores = self.GetModelPerformanceScores()
        self.dataHelper.consoleHelper.PrintTitle("Performance Scores", ConsoleHelper.VERBOSEREQUESTED)
        self.dataHelper.consoleHelper.Display(scores, ConsoleHelper.VERBOSEREQUESTED)


    def GetModelPerformanceScores(self):
        """
        Calculate performance metrics.  Threshold for a positive result can be specified.

        Parameters
        ----------
        threshold : float
            Threshold for classifying the observation success.

        Returns
        -------
        dataFrame : DataFrame
            DataFrame that contains various performance scores for the training and test data.
        """
        # Make sure the model has been initiated and of the correct type.
        if self.model == None:
            raise Exception("The regression model has not be initiated.")

        if len(self.yTrainingPredicted) == 0:
            raise Exception("The predicted values have not been calculated.")

        # Calculate scores.
        # Accuracy.
        trainingScore   = metrics.accuracy_score(self.yTrainingData, self.yTrainingPredicted)
        testScore       = metrics.accuracy_score(self.yTestingData, self.yTestingPredicted)
        accuracyScores  = [trainingScore, testScore]

        # Recall.
        trainingScore   = metrics.recall_score(self.yTrainingData, self.yTrainingPredicted)
        testScore       = metrics.recall_score(self.yTestingData, self.yTestingPredicted)
        recallScores    = [trainingScore, testScore]

        # Precision.
        trainingScore   = metrics.precision_score(self.yTrainingData, self.yTrainingPredicted, zero_division=0)
        testScore       = metrics.precision_score(self.yTestingData, self.yTestingPredicted, zero_division=0)
        precisionScores = [trainingScore, testScore]

        # F1.
        trainingScore   = metrics.f1_score(self.yTrainingData, self.yTrainingPredicted)
        testScore       = metrics.f1_score(self.yTestingData, self.yTestingPredicted)
        f1Scores        = [trainingScore, testScore]


        # Create a DataFrame for returning the values.
        dataFrame = pd.DataFrame({"Accuracy"  : accuracyScores,
                                  "Recall"    : recallScores,
                                  "Precision" : precisionScores,
                                  "F1"        : f1Scores},
                                 index=["Training", "Testing"])

        return dataFrame