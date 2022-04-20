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

    def __init__(self, dataHelper, description=""):
        """
        Constructor.

        Parameters
        ----------
        dataHelper : DataHelper
            DataHelper that has the data in a pandas.DataFrame.
        description : string
            A description of the model.

        Returns
        -------
        None.
        """
        super().__init__(dataHelper, description)


    def CreateModel(self, **kwargs):
        """
        Creates a linear regression model.  Splits the data and creates the model.

        Parameters
        ----------
        **kwargs : keyword arguments
            These arguments are passed on to the model.

        Returns
        -------
        None.
        """
        if len(self.dataHelper.xTrainingData) == 0:
            raise Exception("The data has not been split.")

        self.model = LinearRegression(**kwargs)
        self.model.fit(self.dataHelper.xTrainingData, self.dataHelper.yTrainingData)


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
        trainingScore  = self.model.score(self.dataHelper.xTrainingData, self.dataHelper.yTrainingData)
        testScore      = self.model.score(self.dataHelper.xTestingData, self.dataHelper.yTestingData)
        rSquaredScores = [trainingScore, testScore]

        # Mean square error.
        trainingScore  = mean_squared_error(self.dataHelper.yTrainingData, self.yTrainingPredicted)
        testScore      = mean_squared_error(self.dataHelper.yTestingData, self.yTestingPredicted)
        mseScores      = [trainingScore, testScore]

        # Root mean square error.
        rmseScores     = [np.sqrt(trainingScore), np.sqrt(testScore)]

        # Mean absolute error.
        trainingScore  = mean_absolute_error(self.dataHelper.yTrainingData, self.yTrainingPredicted)
        testScore      = mean_absolute_error(self.dataHelper.yTestingData, self.yTestingPredicted)
        maeScores      = [trainingScore, testScore]

        dataFrame      = pd.DataFrame({"R Squared" : rSquaredScores,
                                       "RMSE"      : rmseScores,
                                       "MSE"       : mseScores,
                                       "MAE"       : maeScores},
                                       index=["Training", "Testing"])
        return dataFrame