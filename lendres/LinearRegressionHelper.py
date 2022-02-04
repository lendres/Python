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
    data            = []

    def __init__(self):
        self.data = []


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
        if not isinstance(self.regressionModel, LinearRegression):
            raise Exception("The regression model has not be initiated.")

        dataFrame = pd.DataFrame(np.append(self.regressionModel.coef_, self.regressionModel.intercept_),
                                 index=self.xTrainingData.columns.tolist()+["Intercept"],
                                 columns=["Coefficients"])
        return dataFrame

        
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
        if not isinstance(self.regressionModel, LinearRegression):
            raise Exception("The regression model has not be initiated.")

        # R squared.
        trainingScore  = self.regressionModel.score(self.xTrainingData, self.yTrainingData)
        testScore      = self.regressionModel.score(self.xTestData, self.yTestData)
        rSquaredScores = [trainingScore, testScore]

        # Mean square error.
        trainingScore  = mean_squared_error(self.yTrainingData, self.regressionModel.predict(self.xTrainingData))
        testScore      = mean_squared_error(self.yTestData, self.regressionModel.predict(self.xTestData))
        mseScores      = [trainingScore, testScore]
        
        # Root mean square error.
        rmseScores     = [np.sqrt(trainingScore), np.sqrt(testScore)]
        
        # Mean absolute error.
        trainingScore  = mean_absolute_error(self.yTrainingData, self.regressionModel.predict(self.xTrainingData))
        testScore      = mean_absolute_error(self.yTestData, self.regressionModel.predict(self.xTestData))
        maeScores      = [trainingScore, testScore]

        dataFrame      = pd.DataFrame({"R Squared" : rSquaredScores,
                                       "RMSE"      : rmseScores,
                                       "MSE"       : mseScores,
                                       "MAE"       : maeScores},
                                       index=["Training", "Test"])
        return dataFrame