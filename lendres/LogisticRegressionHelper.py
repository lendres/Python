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
from sklearn.linear_model import LogisticRegression

import lendres
from lendres.ModelHelper import ModelHelper

class LogisticRegressionHelper(ModelHelper):


    def __init__(self, data):
        super().__init__(data)


    def ConvertCategoryToNumeric(self, column, trueValue):
        if len(self.data) == 0:
            raise Exception("The data has not been set.")

        newColumn               = column + "_int"
        self.data[newColumn]    = 0

        self.additionalDroppedColumns.append(column)

        self.data.loc[self.data[column] == trueValue, newColumn] = 1
        return newColumn


    def CreateModel(self):
        """
        Creates a linear regression model.  Splits the data and creates the model.

        Parameters
        ----------
        data : pandas.DataFrame
            Data in a pandas.DataFrame
        dependentVariable : string
            Name of the column that has the dependant data.
        testSize : double
            Fraction of the data to use as test data.  Must be in the range of 0-1.

        Returns
        -------
        data : pandas.DataFrame
            Data in a pandas.DataFrame
        """

        if len(self.xTrainingData) == 0:
            raise Exception("The data has not been split.")

        self.model = LogisticRegression(solver="liblinear", random_state=1)
        self.model.fit(self.xTrainingData, self.yTrainingData)


    def PlotConfusionMatrix(self, dataSet="training", scale=1.0):
        """
        Plots the confusion matrix for the model output.

        Parameters
        ----------
        dataSet : string
            Which data set(s) to plot.
            training - Plots the results from the training data.
            test     - Plots the results from the test data.
        scale : double
            Scaling parameter used to adjust the plot fonts, lineweights, et cetera for the output scale of the plot.

        Returns
        -------
        None.
        """
        # Initialize to nothing.
        confusionMatrix = None

        # Get the confusion matrix for the correct data set.
        if dataSet == "training":
            if len(self.yTrainingPredicted) == 0:
                self.Predict()
            confusionMatrix = metrics.confusion_matrix(self.yTrainingData, self.yTrainingPredicted)

        elif dataSet == "test":
            if len(self.yTestPredicted) == 0:
                self.Predict()
            confusionMatrix = metrics.confusion_matrix(self.yTestData, self.yTestPredicted)
        else:
            raise Exception("Invalid data set specified.")

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
        plt.figure(figsize=(7,5))

        # Create plot and set the titles.
        axis = sns.heatmap(confusionMatrix, annot=labels, annot_kws={"fontsize" : 12*scale}, fmt="")
        axis.set(title=dataSet.title()+" Data", ylabel="Actual", xlabel="Predicted")

        plt.show()


    def GetModelPerformanceScores(self):
        """
        Calculates the model's scores for the split data (training and testing).

        Parameters
        ----------
        None.

        Returns
        -------
        DataFrame that contains various performance scores for the training and test data.
        """
        # Make sure the model has been initiated and of the correct type.
        if not isinstance(self.model, LogisticRegression):
            raise Exception("The regression model has not be initiated.")

        # R squared.
        trainingScore  = self.model.score(self.xTrainingData, self.yTrainingData)
        testScore      = self.model.score(self.xTestData, self.yTestData)
        rSquaredScores = [trainingScore, testScore]

        # Mean square error.
        trainingScore  = mean_squared_error(self.yTrainingData, self.yTrainingPredicted)
        testScore      = mean_squared_error(self.yTestData, self.yTestPredicted)
        mseScores      = [trainingScore, testScore]

        # Root mean square error.
        rmseScores     = [np.sqrt(trainingScore), np.sqrt(testScore)]

        # Mean absolute error.
        trainingScore  = mean_absolute_error(self.yTrainingData, self.yTrainingPredicted)
        testScore      = mean_absolute_error(self.yTestData, self.yTestPredicted)
        maeScores      = [trainingScore, testScore]

        dataFrame      = pd.DataFrame({"R Squared" : rSquaredScores,
                                       "RMSE"      : rmseScores,
                                       "MSE"       : mseScores,
                                       "MAE"       : maeScores},
                                       index=["Training", "Test"])
        return dataFrame



    def PredictProbabilities(self):
        """
        Runs the probability prediction (model.predict_proba) on the training and test data.  The results are stored in
        the yTrainingPredicted and yTestPredicted variables.

        Parameters
        ----------
        None.

        Returns
        -------
        None.
        """
        # Predict on test
        self.yTrainingPredicted   = self.model.predict_proba(self.xTrainingData)
        self.yTestPredicted       = self.model.predict_proba(self.xTestData)


    def PredictWithThreashold(self):
        """
        Runs the probability prediction (model.predict_proba) on the training and test data.  The results are stored in
        the yTrainingPredicted and yTestPredicted variables.

        Parameters
        ----------
        None.

        Returns
        -------
        None.
        """
        # Predict on test
        self.yTrainingPredicted   = self.model.predict_proba(self.xTrainingData)
        self.yTestPredicted       = self.model.predict_proba(self.xTestData)


    def model_performance_classification_sklearn_with_threshold(self, predictors, target, threshold=0.5):
        """
        Calculate performance metrics.  Threshold for a positive result can be specified.

        Parameters
        ----------
        predictors : independent variables
        target : dependent variable
        threshold : threshold for classifying the observation as class 1

        Returns
        -------
        DataFrame that contains various performance scores for the training and test data.
        """
        # Make sure the model has been initiated and of the correct type.
        if not isinstance(self.model, LogisticRegression):
            raise Exception("The regression model has not be initiated.")

        # predicting using the independent variables
        pred_prob = self.model.predict_proba(predictors)[:, 1]

        pred_thres = pred_prob > threshold
        pred = np.round(pred_thres)

        # Calculate scores.
        accuracyScore   = metrics.accuracy_score(target, pred)
        recallScore     = metrics.recall_score(target, pred)
        precisionScore  = metrics.precision_score(target, pred)
        f1Score         = metrics.f1_score(target, pred)

        # Create a DataFrame for returning the values.
        dataFrame = pd.DataFrame({"Accuracy"  : accuracyScore,
                                  "Recall"    : recallScore,
                                  "Precision" : precisionScore,
                                  "F1"        : f1Score},
                                 index=[0])

        return dataFrame