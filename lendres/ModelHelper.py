"""
Created on January 19, 2022.
@author: Lance
"""
import pandas as pd
import numpy as np

from lendres.ConsoleHelper import ConsoleHelper

class ModelHelper:

    savedModelHelpers         = {}


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
        self.dataHelper                = dataHelper

        self.yTrainingPredicted        = []
        self.yValidationPredicted      = []
        self.yTestingPredicted         = []

        self.model                     = None

        # Features for comparing models.
        self.description               = description


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
        self.dataHelper                = original.dataHelper.Copy(deep=deep)


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
            if len(modelHelpers[0].dataHelper.xValidationData) != 0:
                columns.append("Validation " + score)
            columns.append("Testing " + score)

        # Initialize the DataFrame to the correct size and add the index and columns.
        comparisonFrame = pd.DataFrame(index=index, columns=columns)

        for modelHelper in modelHelpers:
            results = modelHelper.GetModelPerformanceScores(final=True)

            for score in scores:
                comparisonFrame.loc[modelHelper.GetName(), "Training "+score] = results.loc["Training", score]
                if len(modelHelper.dataHelper.xValidationData) != 0:
                    comparisonFrame.loc[modelHelper.GetName(), "Validation "+score]  = results.loc["Validation", score]
                comparisonFrame.loc[modelHelper.GetName(), "Testing "+score]  = results.loc["Testing", score]

        return comparisonFrame


    @classmethod
    def PrintModelComparisons(cls, scores, modelHelpers=None):
        """
        Prints the model comparisons.

        Parameters
        ----------
        scores : string or list of strings.
            The score or scores to print out.
        modelHelpers : list of ModelHelpers
            A list of ModelHelpers to get and print the scores of.  If none is supplied, the saved list
            of MOdelHelpers is used.

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
        self.yTrainingPredicted        = self.model.predict(self.dataHelper.xTrainingData)
        self.yTestingPredicted         = self.model.predict(self.dataHelper.xTestingData)

        if len(self.dataHelper.yValidationData) != 0:
            self.yValidationPredicted  = self.model.predict(self.dataHelper.xValidationData)


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
                                 index=self.dataHelper.xTrainingData.columns.tolist()+["Intercept"],
                                 columns=["Coefficients"])
        return dataFrame