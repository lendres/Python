# -*- coding: utf-8 -*-
"""
Created on Wed Jan 19 07:49:25 2022

@author: Lance
"""

from IPython.display import display
import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

#import lendres

class LinearRegressionHelper:
    xTrainingData           = []
    xTestData               = []
    yTrainingData           = []
    yTestData               = []
    regressionModel         = []
#    data            = []

#    def __init__(self):
#        self.data = []


    def CreateLinearModel(self, data, dependentVariable, testSize):
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

        # Remove the dependent varaible from the rest of the data.
        x = data.drop([dependentVariable], axis=1)

        # The dependent variable.
        y = data[[dependentVariable]]

        # Split the data.
        self.xTrainingData, self.xTestData, self.yTrainingData, self.yTestData = train_test_split(x, y, test_size=testSize, random_state=1)

        self.regressionModel = LinearRegression()
        self.regressionModel.fit(self.xTrainingData, self.yTrainingData)



    def DisplayModelCoefficients(self):
        """
        Displays the coefficients and intercept of a linear regression model.

        Parameters
        ----------
        None.

        Returns
        -------
        None.
        """
        dataFrame = pd.DataFrame(np.append(self.regressionModel.coef_, self.regressionModel.intercept_),
                                 index = self.xTrainingData.columns.tolist() + ["Intercept"],
                                 columns = ["Coefficients"])
        display(dataFrame)