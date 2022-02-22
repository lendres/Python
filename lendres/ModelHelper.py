# -*- coding: utf-8 -*-
"""
Created on Wed Jan 19 07:49:25 2022

@author: Lance
"""

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

class ModelHelper:

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
        self.additionalDroppedColumns  = []
        self.data                      = data

        self.xTrainingData             = []
        self.xTestingData              = []
        self.yTrainingData             = []
        self.yTestingData              = []

        self.model                     = None

        self.yTrainingPredicted        = []
        self.yTestingPredicted         = []
        
    
    def CopyData(self, original, deep=False):
        self.additionalDroppedColumns  = original.additionalDroppedColumns.copy()
        self.data                      = original.data.copy(deep=deep)

        self.xTrainingData             = original.xTrainingData.copy(deep=deep)
        self.xTestingData              = original.xTestingData.copy(deep=deep)
        self.yTrainingData             = original.yTrainingData.copy(deep=deep)
        self.yTestingData              = original.yTestingData.copy(deep=deep)


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
        data : pandas.DataFrame
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
        self.xTrainingData, self.xTestingData, self.yTrainingData, self.yTestingData = train_test_split(x, y, test_size=testSize, random_state=1)


    def GetSplitComparisons(self):
        """
        Returns the value counts and percentages of the dependant variable for the
        original, training, and testing data.

        Returns
        -------
        comparisonFrame : pandas.DataFrame
            DataFrame with the counts and percentages.
        """
        comparisonFrame = pd.DataFrame(
                    {"False"    : [self.GetCountAndPrecentString(0, "original"),
                                   self.GetCountAndPrecentString(0, "training"),
                                   self.GetCountAndPrecentString(0, "testing")],
                     "Positive" : [self.GetCountAndPrecentString(1, "original"),
                                   self.GetCountAndPrecentString(1, "training"),
                                   self.GetCountAndPrecentString(1, "testing")]},
                     index=["Original", "Training", "Testing"])
        return comparisonFrame


    def GetCountAndPrecentString(self, classValue, dataSet="original", column=None):
        """
        Gets a string that is the value count of "classValue" and the percentage of the total
        that the "classValue" accounts for in the column.

        Parameters
        ----------
        classValue : int or string
            The value to count and calculate the percentage for.
        dataSet : string
            Which data set(s) to plot.
            original - Gets the results from the original data.
            training - Gets the results from the training data.
            testing  - v the results from the test data.
        column : string, optional
            If provided, that column will be used for the calculations.  If no value is provided,
            the dependant variable is used. The default is None.

        Returns
        -------
        None.

        """
        data = None

        if dataSet == "original":
            data = self.data
        elif dataSet == "training":
            data = self.yTrainingData
        elif dataSet == "testing":
            data = self.yTestingData
        else:
            raise Exception("Invalid data set specified.")

        if column == None:
            column = self.GetDependentVariableName()

        classValueCount = len(data.loc[data[column] == classValue])

        string = "{0} ({1:0.2f}%)".format(classValueCount, classValueCount/len(data.index) * 100)
        return string

        
    def GetDependentVariableName(self):
        """
        Returns the name of the dependent variable (column heading).

        Parameters
        ----------
        None.

        Returns
        -------
        : string
            The name (column heading) of the dependent variable.

        """
        if len(self.yTrainingData) == 0:
            raise Exception("The data has not been split (dependent variable not set).")
        
        return self.yTrainingData.columns[0]


    def Predict(self):
        """
        Runs the prediction (model.predict) on the training and test data.  The results are stored in
        the yTrainingPredicted and yTestingPredicted variables.

        Parameters
        ----------
        None.

        Returns
        -------
        None.
        """
        # Predict on the training and testing data.
        self.yTrainingPredicted   = self.model.predict(self.xTrainingData)
        self.yTestingPredicted    = self.model.predict(self.xTestingData)


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