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

class ModelHelper:

    def __init__(self, data):
        self.additionalDroppedColumns  = []
        self.data                      = data

        self.xTrainingData             = []
        self.xTestData                 = []
        self.yTrainingData             = []
        self.yTestData                 = []

        self.model                     = None

        self.yTrainingPredicted        = []
        self.yTestPredicted            = []


    def EncodeAllCategoricalColumns(self, data):
        """
        Converts all categorical columns (have that data type "category") to one hot encoded values and drops one
        value per category.  Prepares categorical columns for use in a model.

        Parameters
        ----------
        data : pandas.DataFrame
            DataFrame to operate on.

        Returns
        -------
        data : DataFrame
            The new DataFrame with the encoded values.
        """
        # Find all the category types in the DataFrame.
        # Gets all the columns that have the category data type.  That is returned as a DataSeries.  The
        # index (where the names are) is extracted from that.
        allCategoricalColumns = data.dtypes[data.dtypes == 'category'].index.tolist()

        return self.EncodeCategoricalColumns(data, allCategoricalColumns)


    def EncodeCategoricalColumns(self, data, categories):
        """
        Converts the categorical columns "categories" to one hot encoded values and drops one value per category.
        Prepares categorical columns for use in a model.

        Parameters
        ----------
        data : pandas.DataFrame
            DataFrame to operate on.

        Returns
        -------
        data : DataFrame
            The new DataFrame with the encoded values.
        """
        return pd.get_dummies(data, columns=categories, drop_first=True)


    def SplitData(self, dependentVariable, testSize):
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
        x = self.data.drop([dependentVariable], axis=1)
        x = x.drop(self.additionalDroppedColumns, axis=1)

        # The dependent variable.
        y = self.data[[dependentVariable]]

        # Split the data.
        self.xTrainingData, self.xTestData, self.yTrainingData, self.yTestData = train_test_split(x, y, test_size=testSize, random_state=1)


    def Predict(self):
        # Predict on test
        self.yTrainingPredicted   = self.model.predict(self.xTrainingData)
        self.yTestPredicted       = self.model.predict(self.xTestData)


    def GetModelCoefficients(self):
        """
        Displays the coefficients and intercept of a linear regression model.

        Parameters
        ----------
        None.

        Returns
        -------
        DataFrame that contains the model coefficients.
        """

        # Make sure the model has been initiated and of the correct type.
        if self.model == None:
            raise Exception("The regression model has not be initiated.")

        dataFrame = pd.DataFrame(np.append(self.model.coef_, self.model.intercept_),
                                 index=self.xTrainingData.columns.tolist()+["Intercept"],
                                 columns=["Coefficients"])
        return dataFrame