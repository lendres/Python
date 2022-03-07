# -*- coding: utf-8 -*-
"""
Created on Wed Jan 19 07:49:25 2022

@author: Lance
"""

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

class ModelHelper:

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
        self.additionalDroppedColumns  = []
        self.dataHelper                = dataHelper

        self.xTrainingData             = []
        self.xTestingData              = []
        self.yTrainingData             = []
        self.yTestingData              = []

        self.model                     = None

        self.yTrainingPredicted        = []
        self.yTestingPredicted         = []


    def CopyData(self, original, deep=False):
        """
        Copies the data from another ModelHelper.  Does not copy any models built
        or output produced.

        Parameters
        ----------
        original : ModelHelper
            The source of the data.
        deep : bool, optional
            Specifies if a deep copy should be done. The default is False.

        Returns
        -------
        None.

        """
        self.additionalDroppedColumns  = original.additionalDroppedColumns.copy()
        self.dataHelper                = original.dataHelper.Copy(deep=deep)

        self.xTrainingData             = original.xTrainingData.copy(deep=deep)
        self.xTestingData              = original.xTestingData.copy(deep=deep)
        self.yTrainingData             = original.yTrainingData.copy(deep=deep)
        self.yTestingData              = original.yTestingData.copy(deep=deep)


    def SplitData(self, dependentVariable, testSize, stratify=True):
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
        x = self.dataHelper.data.drop([dependentVariable], axis=1)
        x = x.drop(self.additionalDroppedColumns, axis=1)

        # The dependent variable.
        y = self.dataHelper.data[dependentVariable]
        if stratify:
            stratify = y
        else:
            stratify = None

        # Split the data.
        self.xTrainingData, self.xTestingData, self.yTrainingData, self.yTestingData = train_test_split(x, y, test_size=testSize, random_state=1, stratify=stratify)


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
            data = self.dataHelper.data
        elif dataSet == "training":
            data = pd.DataFrame(self.yTrainingData)
        elif dataSet == "testing":
            data = pd.DataFrame(self.yTestingData)
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

        return self.yTrainingData.name


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