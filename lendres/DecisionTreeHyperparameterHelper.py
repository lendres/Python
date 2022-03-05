# -*- coding: utf-8 -*-
"""
Created on Wed Jan 19 07:49:25 2022

@author: Lance
"""
from IPython.display import display

from sklearn import metrics
from sklearn.model_selection import GridSearchCV
import lendres
from lendres.DecisionTreeHelper import DecisionTreeHelper

class DecisionTreeHyperparameterHelper(DecisionTreeHelper):

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
        super().__init__(dataHelper)
        self.gridSearch              = None


    @classmethod
    def FromData(cls, original, deep=False):
        """
        Creates a new DecisionTreeHyperparameterHelper by copying the data from the original.

        Parameters
        ----------
        original : DecisionTreeHyperparameterHelper
            The source instance to copy from.
        deep : bool, optional
            DESCRIPSpecifies if a deep copy should be done. The default is False.

        Returns
        -------
        decisionTreeHelper : DecisionTreeHyperparameterHelper
            Returns a new DecisionTreeHyperparameterHelper based on data copied from the original.

        """
        decisionTreeHelper = DecisionTreeHyperparameterHelper(None)
        decisionTreeHelper.CopyData(original, deep)
        return decisionTreeHelper


    def CreateGridSearchModel(self, parameters, scoringFunction, displayBestParameters=True, **kwargs):
        """
        Creates a cross validation grid search model.

        Parameters
        ----------
        parameters : dictionary
            Grid search parameters.
        scoringFunction : function
            Method use to calculate a score for the model.
        displayBestParameters : bool
            If true, it outputs the parameters of the selected model.
        **kwargs : keyword arguments
            These arguments are passed on to the GridSearchCV.

        Returns
        -------
        None.

        """
        # Make sure there is data to operate on.
        if len(self.xTrainingData) == 0:
            raise Exception("The data has not been split.")

        # Type of scoring used to compare parameter combinations.
        scorer = metrics.make_scorer(scoringFunction)

        # Run the grid search.
        self.gridSearch = GridSearchCV(self.model, parameters, scoring=scorer, **kwargs)
        self.gridSearch = self.gridSearch.fit(self.xTrainingData, self.yTrainingData)

        # Set the model (classifier) to the best combination of parameters.
        self.model = self.gridSearch.best_estimator_

        # Fit the best algorithm to the data.
        self.model.fit(self.xTrainingData, self.yTrainingData)


    def DisplayChosenParameters(self):
        """
        Prints the model performance scores.

        Parameters
        ----------
        useMarkDown : bool
            If true, markdown output is enabled.

        Returns
        -------
        None.
        """
        self.dataHelper.consoleHelper.PrintBold("Chosen Model Parameters")
        display(self.gridSearch.best_params_)