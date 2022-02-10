# -*- coding: utf-8 -*-
"""
Created on Wed Jan 19 07:49:25 2022

@author: Lance
"""

import pandas as pd
import numpy as np

from sklearn import metrics
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

from lendres.ModelHelper import ModelHelper

class LogisticRegressionHelper(ModelHelper):

    #def __init__(self):
        #super().__init__()


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
        trainingScore  = mean_squared_error(self.yTrainingData, self.model.predict(self.xTrainingData))
        testScore      = mean_squared_error(self.yTestData, self.model.predict(self.xTestData))
        mseScores      = [trainingScore, testScore]

        # Root mean square error.
        rmseScores     = [np.sqrt(trainingScore), np.sqrt(testScore)]

        # Mean absolute error.
        trainingScore  = mean_absolute_error(self.yTrainingData, self.model.predict(self.xTrainingData))
        testScore      = mean_absolute_error(self.yTestData, self.model.predict(self.xTestData))
        maeScores      = [trainingScore, testScore]

        dataFrame      = pd.DataFrame({"R Squared" : rSquaredScores,
                                       "RMSE"      : rmseScores,
                                       "MSE"       : mseScores,
                                       "MAE"       : maeScores},
                                       index=["Training", "Test"])
        return dataFrame