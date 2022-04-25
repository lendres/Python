"""
Created on Wed January 19, 2022
@author: Lance
"""

import numpy                      as np
import matplotlib.pyplot          as plt

from lendres.PlotHelper           import PlotHelper
from lendres.DecisionTreeHelper   import DecisionTreeHelper

class DecisionTreeCostComplexityHelper(DecisionTreeHelper):

    def __init__(self, dataHelper, model=None, description=""):
        """
        Constructor.

        Parameters
        ----------
        dataHelper : DataHelper
            DataHelper that has the data in a pandas.DataFrame.
        model : Model
            A regression model.
        description : string
            A description of the model.

        Returns
        -------
        None.
        """
        super().__init__(dataHelper, model, description)
        self.costComplexityPath                 = None
        self.DecisionTreeCostComplexityHelpers  = None


    @classmethod
    def FromData(cls, original, deep=False, **kwargs):
        """
        Creates a new DecisionTreeCostComplexityHelper by copying the data from the original.

        Parameters
        ----------
        original : DecisionTreeCostComplexityHelper
            The source instance to copy from.
        deep : bool, optional
            Specifies if a deep copy should be done. The default is False.
        **kwargs : keyword arguments
            These arguments are passed on to the DecisionTreeClassifier.

        Returns
        -------
        decisionTreeHelper : DecisionTreeCostComplexityHelper
            Returns a new DecisionTreeCostComplexityHelper based on data copied from the original.

        """
        decisionTreeHelper = DecisionTreeCostComplexityHelper(None, DecisionTreeHelper.CreateDefaultModel(**kwargs))
        decisionTreeHelper.CopyData(original, deep)
        return decisionTreeHelper


    def CreateCostComplexityPruningModel(self, criteria):
        """
        Creates a cost complexity pruning model.

        Parameters
        ----------
        criteria : string
            Critera used to score the models.  The options are:
                "accuracy"
                "recall"
                "precision"
                "f1"

        Returns
        -------
        None.
        """
        # Build the path.
        self.costComplexityPath  = self.model.cost_complexity_pruning_path(self.dataHelper.xTrainingData, self.dataHelper.yTrainingData)

        # Get all the alphas except the trivial case (the case with one node).
        ccpAlphas                = self.costComplexityPath.ccp_alphas[:-1]
        self.decisionTreeHelpers = []

        # Create models based on the cost complexity pruning alpha values.
        for ccpAlpha in ccpAlphas:
            decisionTreeHelper = DecisionTreeCostComplexityHelper.FromData(self, deep=False, ccp_alpha=ccpAlpha)
            decisionTreeHelper.Fit()
            self.decisionTreeHelpers.append(decisionTreeHelper)

        # Calculate the scores and use them to select the best model.  The model is
        # stored in the standard model location.
        trainingScores, testScores = self.GetCostComplexityPruningScores(criteria)
        bestModelIndex             = np.argmax(testScores)
        self.model                 = self.decisionTreeHelpers[bestModelIndex].model


    def GetCostComplexityPruningScores(self, criteria):
        """
        Loops through all the models created from a cost complexity pruning decision
        tree and gets all the training and test scores.

        Parameters
        ----------
        criteria : string
            Critera used to score the models.  The options are:
                "accuracy"
                "recall"
                "precision"
                "f1"

        Returns
        -------
        trainingScores : float
            A list of all the training scores from the models.
        testScores : float
            A list of all the testing scores from the models.
        """
        # Converts the criteria into title case which is what is required to extract
        # the scores from the DataFrame that contains all available scores.
        criteriaName = criteria.title()

        trainingScores = []
        testScores     = []

        for decisionTreeHelper in self.decisionTreeHelpers:
            # Predict the dependent variable results and extract the test scores.
            decisionTreeHelper.Predict()
            performanceScores = decisionTreeHelper.GetModelPerformanceScores(final=True)

            # The test scores are returned in a DataFrame with all available test scores.
            # Here we extract just the scores for the specific criteria we are using.
            trainingScores.append(performanceScores.loc["Training", criteriaName])
            testScores.append(performanceScores.loc["Testing", criteriaName])

        return trainingScores, testScores


    def CreateAlphasVersusScoresPlot(self, criteria):
        """
        Plots the alphas versus training and/or testing scores for all the models
        generated from a cost complexity pruning model.

        Parameters
        ----------
        criteria : string
            Critera used to score the models.  The options are:
                "accuracy"
                "recall"
                "precision"
                "f1"

        Returns
        -------
        None.
        """
        # Get the data for plotting.  We don't use the last alpha which is the trivial
        # case (single node).
        trainingScores, testScores = self.GetCostComplexityPruningScores(criteria)
        ccpAlphas                  = self.costComplexityPath.ccp_alphas[:-1]

        # Must be run before creating figure or plotting data.
        PlotHelper.FormatPlot()
        axis = plt.gca()

        # The actual plotting part.
        axis.plot(ccpAlphas, trainingScores, marker='o', label="Training", drawstyle="steps-post", color="#1f77b4")
        axis.plot(ccpAlphas, testScores, marker='o', label="Testing", drawstyle="steps-post", color="#ff7f0e")

        # Gussy up this critter with some titles and a legend.
        criteriaName = criteria.title()
        axis.set(title=criteriaName+" vs Alpha", xlabel="Alpha", ylabel=criteriaName)
        axis.legend()

        plt.show()


    def CreateImpunityVersusAlphaPlot(self):
        """
        Creates an impunity versus alpha plot.

        Parameters
        ----------
        None.

        Returns
        -------
        None.
        """
        # Get the data for plotting.
        ccpAlphas  = self.costComplexityPath.ccp_alphas[:-1]
        impurities = self.costComplexityPath.impurities[:-1]

        # Must be run before creating figure or plotting data.
        PlotHelper.FormatPlot()

        axis = plt.gca()
        axis.plot(ccpAlphas, impurities, marker='o', drawstyle="steps-post")
        axis.set(title="Total Impurity vs Effective Alpha\nTraing Data", xlabel="Effective Alpha", ylabel="Total Impurity of Leaves")

        plt.show()