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

    savedModelHelpers         = {}


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
        self.xValidationData           = []
        self.yValidationData           = []
        self.yTrainingData             = []
        self.yTestingData              = []

        self.model                     = None

        self.yTrainingPredicted        = []
        self.yTestingPredicted         = []

        # Features for comparing models.
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
    def SaveModelHelper(cls, modelHelper):
        """
        Saves the ModelHelper in a dictionary.  The ModelHelpers are indexed by their name as defined
        by the GetName function.

        ModelHelpers with the same name cannot be saved, they will over write each other.  This is
        by design, so that a test can be run mulple time and be tuned while ModelHelper only saves
        the last version passed to it.

        Parameters
        ----------
        modelHelper : ModelHelper
            ModelHelper to save.

        Returns
        -------
        None.
        """
        cls.savedModelHelpers[modelHelper.GetName()] = modelHelper


    @classmethod
    def GetModelComparisons(cls, scores, modelHelpers=None):
        """
        Creates a comparison of a score across several models.

        Extracts the training and testing "score" for each model and puts it into a DataFrame.  The models must
        all have a "GetModelPerformanceScores" function that returns a DataFrame that contains a "score" column
        and "Testing" and "Training" rows.

        If the model has a model.description and it is not blank, that value is used for the name of the model,
        otherwise, the class name of the model is used.

        Parameters
        ----------
        modelHelpers : list or dictionary
            List of ModelHelpers to compare.  If none is supplied, the list is taken from those stored in
            the ModelHelper.
        score : string or list of strings
            Score to extract from the list of performance scores.  This must be a column in the DataFrame
            returned by the "GetModelPerformanceScores" function.

        Returns
        -------
        comparisonFrame : pandas.DataFrame
            A DataFrame that contains a list of the models and each models training and testing score.
        """
        # Ensure scores is a list.
        if type(scores) != list:
            scores = [scores]

        # Get a list of the ModelHelpers.
        if modelHelpers == None:
            modelHelpers = ModelHelper.savedModelHelpers

        # If it is a dictionary, flatten it into a list for easier use.
        if type(modelHelpers) == dict:
            modelHelpers = list(modelHelpers.values())

        # Create the index for the DataFrame.
        index   = []
        for modelHelper in modelHelpers:
            index.append(modelHelper.GetName())

        # Create the column names for the DataFrame.
        columns = []
        for score in scores:
            columns.append("Training " + score)
            columns.append("Testing " + score)

        # Initialize the DataFrame to the correct size and add the index and columns.
        comparisonFrame = pd.DataFrame(index=index, columns=columns)

        for modelHelper in modelHelpers:
            results = modelHelper.GetModelPerformanceScores()

            for score in scores:
                comparisonFrame.loc[modelHelper.GetName(), "Training "+score] = results.loc["Training", score]
                comparisonFrame.loc[modelHelper.GetName(), "Testing "+score]  = results.loc["Testing", score]

        return comparisonFrame


    @classmethod
    def PrintModelComparisons(cls, scores, modelHelpers=None):
        """
        Prints the model comparisons.

        Parameters
        ----------
        modelHelpers : list of ModelHelpers
            A list of ModelHelpers to get and print the scores of.  If none is supplied, the saved list
            of MOdelHelpers is used.
        scores : string or list of strings.
            The score or scores to print out.

        Returns
        -------
        None.
        """
        comparisons = cls.GetModelComparisons(scores, modelHelpers)

        modelHelper = None
        if modelHelpers == None:
            modelHelper = list(ModelHelper.savedModelHelpers.values())[0]
        else:
            modelHelper = modelHelpers[0]

        modelHelper.dataHelper.consoleHelper.Display(comparisons, ConsoleHelper.VERBOSEREQUESTED)


    def GetName(self):
        """
        Gets the name of the model.  If a description has been provided, that is used.  Otherwise,
        the name of the class is returned.

        Returns
        -------
        : string
            A string representing the name of the model.
        """
        # The name of the model to use as the index.  Use the more useful "description"
        # if it is available, otherwise use the calls name.
        if self.description == "":
            return self.__class__.__name__
        else:
            return self.description


    def PrintClassName(self):
        """
        Displays the class name according to the behavior specified in the ConsoleHelper.

        Returns
        -------
        None.
        """
        self.dataHelper.consoleHelper.Print(self.__class__.__name__, ConsoleHelper.VERBOSEREQUESTED)


    def SplitData(self, dependentVariable, testSize, validationSize=None, stratify=False):
        """
        Creates a linear regression model.  Splits the data and creates the model.

        Parameters
        ----------
        dependentVariable : string
            Name of the column that has the dependant data.
        testSize : double
            Fraction of the data to use as test data.  Must be in the range of 0-1.
        validationSize : double
            Fraction of the non-test data to use as validation data.  Must be in the range of 0-1.
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

        if validationSize != None:
            self.xTrainingData, self.xValidationData, self.yTrainingData, self.yValidationData = train_test_split(self.xTrainingData, self.yTrainingData, test_size=validationSize, random_state=1, stratify=stratify)



    def GetSplitComparisons(self):
        """
        Returns the value counts and percentages of the dependant variable for the
        original, training (if available), and testing (if available) data.

        Returns
        -------
        comparisonFrame : pandas.DataFrame
            DataFrame with the counts and percentages.
        """
        # Get results for original data.
        false = [self.GetCountAndPrecentString(0, "original")]
        true  = [self.GetCountAndPrecentString(1, "original")]
        index = ["Original"]


        # If the data has been split, we will add the split information as well.
        if len(self.xTrainingData) != 0:
            false.append(self.GetCountAndPrecentString(0, "training"))
            true.append(self.GetCountAndPrecentString(1, "training"))
            index.append("Training")

            if len(self.xValidationData) != 0:
                false.append(self.GetCountAndPrecentString(0, "validation"))
                true.append(self.GetCountAndPrecentString(1, "validation"))
                index.append("Validation")

            false.append(self.GetCountAndPrecentString(0, "testing"))
            true.append(self.GetCountAndPrecentString(1, "testing"))
            index.append("Testing")

        # Create the data frame.
        comparisonFrame = pd.DataFrame(
                    {"False" : false,
                     "True"  : true},
                     index=index)

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

        elif dataSet == "validation":
            classValueCount = sum(self.yValidationData == classValue)
            totalCount      = self.yValidationData.size

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