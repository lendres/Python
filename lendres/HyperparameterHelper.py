"""
Created on January 19, 2022
@author: Lance
"""
#from IPython.display import display

from sklearn                   import metrics
from sklearn.model_selection   import GridSearchCV
from sklearn.model_selection   import RandomizedSearchCV

from lendres.ConsoleHelper import ConsoleHelper

class HyperparameterHelper():

    def __init__(self, categoricalHelper):
        """
        Constructor.

        Parameters
        ----------
        categoricalHelper : CategoricalHelper
            CategoricalHelper used in the grid search.

        Returns
        -------
        None.
        """
        self.categoricalHelper    = categoricalHelper
        self.searcher             = None


    def CreateSearchModel(self, which, parameters, scoringFunction, **kwargs):
        """
        Creates a cross validation grid search model.

        Parameters
        ----------
        which : string
            Type of search to perform.
            random : Random search.
            grid : Grid search.
        parameters : dictionary
            Search parameters.
        scoringFunction : function
            Method use to calculate a score for the model.
        **kwargs : keyword arguments
            These arguments are passed on to the GridSearchCV.

        Returns
        -------
        None.

        """
        # Make sure there is data to operate on.
        if len(self.categoricalHelper.dataHelper.xTrainingData) == 0:
            raise Exception("The data has not been split.")

        # Type of scoring used to compare parameter combinations.
        scorer = metrics.make_scorer(scoringFunction)

        # Run the grid search.
        if which == "grid":
            self.searcher = GridSearchCV(self.categoricalHelper.model, parameters, scoring=scorer, **kwargs)
        elif which == "random":
            self.searcher = RandomizedSearchCV(self.categoricalHelper.model, parameters, scoring=scorer, **kwargs)

        self.searcher = self.searcher.fit(self.categoricalHelper.dataHelper.xTrainingData, self.categoricalHelper.dataHelper.yTrainingData)

        # Set the model (categoricalHelper) to the best combination of parameters.
        self.categoricalHelper.model = self.searcher.best_estimator_

        # Fit the best algorithm to the data.
        self.categoricalHelper.model.fit(self.categoricalHelper.dataHelper.xTrainingData, self.categoricalHelper.dataHelper.yTrainingData)


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
        self.categoricalHelper.dataHelper.consoleHelper.PrintBold("Chosen Model Parameters", ConsoleHelper.VERBOSEREQUESTED)
        self.categoricalHelper.dataHelper.consoleHelper.Display(self.searcher.best_params_, ConsoleHelper.VERBOSEREQUESTED)