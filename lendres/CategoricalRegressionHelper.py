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

import lendres
from lendres.ModelHelper import ModelHelper

class CategoricalRegressionHelper(ModelHelper):

    def __init__(self, data):
        """
        Constructor.
        
        Parameters
        ----------
        data : pandas.DataFrame
            DataFrame to operate on.
            
        Returns
        -------
        None.
        """
        super().__init__(data)


    def ConvertCategoryToNumeric(self, column, trueValue):
        if len(self.data) == 0:
            raise Exception("The data has not been set.")

        newColumn               = column + "_int"
        self.data[newColumn]    = 0

        self.additionalDroppedColumns.append(column)

        self.data.loc[self.data[column] == trueValue, newColumn] = 1
        return newColumn


    def CreateConfusionMatrixPlot(self, dataSet="training", scale=1.0):
        """
        Plots the confusion matrix for the model output.

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
        scale *= 1.5
        lendres.Plotting.FormatPlot(scale=scale)

        # Not much is shown, so we can shrink the figure size.
        plt.figure(figsize=(9,6))

        # Create plot and set the titles.
        axis = sns.heatmap(confusionMatrix, annot=labels, annot_kws={"fontsize" : 14*scale}, fmt="")
        axis.set(title=dataSet.title()+" Data", ylabel="Actual", xlabel="Predicted")

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


    def GetModelPerformanceScores(self, threshold=0.5):
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
        trainingScore   = metrics.accuracy_score(self.yTrainingData, self.yTrainingPredicted)
        testScore       = metrics.accuracy_score(self.yTestingData, self.yTestingPredicted)
        accuracyScores  = [trainingScore, testScore]

        trainingScore   = metrics.recall_score(self.yTrainingData, self.yTrainingPredicted)
        testScore       = metrics.recall_score(self.yTestingData, self.yTestingPredicted)
        recallScores    = [trainingScore, testScore]

        trainingScore   = metrics.precision_score(self.yTrainingData, self.yTrainingPredicted)
        testScore       = metrics.precision_score(self.yTestingData, self.yTestingPredicted)
        precisionScores = [trainingScore, testScore]

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