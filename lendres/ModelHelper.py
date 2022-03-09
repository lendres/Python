# -*- coding: utf-8 -*-
"""
Created on Wed Jan 19 07:49:25 2022

@author: Lance
"""

import pandas as pd
import numpy as np

from lendres.ConsoleHelper import ConsoleHelper

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

        self.description               = ""


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


    @classmethod
    def GetModelComparisons(cls, models, score):
        """
        Creates a comparison of a score across several models.

        Extracts the training and testing "score" for each model and puts it into a DataFrame.  The models must
        all have a "GetModelPerformanceScores" function that returns a DataFrame that contains a "score" column
        and "Testing" and "Training" rows.

        If the model has a model.description and it is not blank, that value is used for the name of the model,
        otherwise, the class name of the model is used.

        Parameters
        ----------
        models : list
            List of models to compare.
        score : string
            Score to extract from the list of performance scores.  This must be a column in the DataFrame
            returned by the "GetModelPerformanceScores" function.

        Returns
        -------
        comparisonFrame : pandas.DataFrame
            A DataFrame that contains a list of the models and each models training and testing score.
        """

        trainingScores = []
        testingScores  = []
        index          = []

        for model in models:
            results = model.GetModelPerformanceScores()
            trainingScores.append(results.loc["Training", score])
            testingScores.append(results.loc["Testing", score])

            # The name of the model to use as the index.  Use the more useful "description"
            # if it is available, otherwise use the calls name.
            if model.description == "":
                index.append(model.__class__.__name__)
            else:
                index.append(model.description)

        comparisonFrame = pd.DataFrame({"Training "+score : trainingScores,
                                        "Testing "+score  : testingScores},
                                        index=index)

        return comparisonFrame


    @classmethod
    def PrintModelComparisons(cls, models, score):
        comparisons = cls.GetModelComparisons(models, score)

        models[0].dataHelper.consoleHelper.Display(comparisons)


    def PrintClassName(self):
        """
        Displays the class name according to the behavior specified in the ConsoleHelper.

        Returns
        -------
        None.
        """
        self.dataHelper.consoleHelper.Print(self.__class__.__name__, ConsoleHelper.VERBOSEREQUESTED)


    def SplitData(self, dependentVariable, testSize, stratify=False):
        """
        Creates a linear regression model.  Splits the data and creates the model.

        Parameters
        ----------
        dependentVariable : string
            Name of the column that has the dependant data.
        testSize : double
            Fraction of the data to use as test data.  Must be in the range of 0-1.
        stratify : bool
            If true, the approximate ratio of value in the dependent variable is maintained.

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
        classValueCount = 0
        totalCount      = 0

        if dataSet == "original":
            if column == None:
                column = self.GetDependentVariableName()
            classValueCount = sum(self.dataHelper.data[column] == classValue)
            totalCount      = self.dataHelper.data[column].size

        elif dataSet == "training":
            classValueCount = sum(self.yTrainingData == classValue)
            totalCount      = self.yTrainingData.size

        elif dataSet == "testing":
            classValueCount = sum(self.yTestingData == classValue)
            totalCount      = self.yTestingData.size

        else:
            raise Exception("Invalid data set specified.")

        string = "{0} ({1:0.2f}%)".format(classValueCount, classValueCount/totalCount * 100)
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