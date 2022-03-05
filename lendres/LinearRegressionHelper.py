# -*- coding: utf-8 -*-
"""
Created on Wed Jan 19 07:49:25 2022

@author: Lance
"""

import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

from lendres.ModelHelper import ModelHelper

class LinearRegressionHelper(ModelHelper):

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

        self.model = LinearRegression()
        self.model.fit(self.xTrainingData, self.yTrainingData)


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
        if not isinstance(self.model, LinearRegression):
            raise Exception("The regression model has not be initiated.")

        # Make sure the predictions have been made on the training and test data.
        if len(self.yTrainingPredicted) == 0:
            self.Predict()

        # R squared.
        trainingScore  = self.model.score(self.xTrainingData, self.yTrainingData)
        testScore      = self.model.score(self.xTestingData, self.yTestingData)
        rSquaredScores = [trainingScore, testScore]

        # Mean square error.
        trainingScore  = mean_squared_error(self.yTrainingData, self.yTrainingPredicted)
        testScore      = mean_squared_error(self.yTestingData, self.yTestingPredicted)
        mseScores      = [trainingScore, testScore]

        # Root mean square error.
        rmseScores     = [np.sqrt(trainingScore), np.sqrt(testScore)]

        # Mean absolute error.
        trainingScore  = mean_absolute_error(self.yTrainingData, self.yTrainingPredicted)
        testScore      = mean_absolute_error(self.yTestingData, self.yTestingPredicted)
        maeScores      = [trainingScore, testScore]

        dataFrame      = pd.DataFrame({"R Squared" : rSquaredScores,
                                       "RMSE"      : rmseScores,
                                       "MSE"       : mseScores,
                                       "MAE"       : maeScores},
                                       index=["Training", "Testing"])
        return dataFrame