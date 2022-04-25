"""
Created on January 19, 2022
@author: Lance
"""
#from IPython.display import display

from sklearn                                 import metrics
from sklearn.model_selection                 import GridSearchCV
from sklearn.model_selection                 import RandomizedSearchCV

from lendres.ConsoleHelper                   import ConsoleHelper
from lendres.ModelHelper                     import ModelHelper
#from lendres.CategoricalRegressionHelper     import CategoricalRegressionHelper

class HyperparameterHelper():

    def __init__(self, categoricalHelper, scoringFunction, searchType="random", testing=False):
        """
        Constructor.

        Parameters
        ----------
        categoricalHelper : CategoricalHelper
            CategoricalHelper used in the grid search.
        scoringFunction : function
            Method use to calculate a score for the model.
        searchType : string
            Type of search to perform.
            random : Random search.
            grid : Grid search.
         tesing : bool
             If true, additional output is enable.

        Returns
        -------
        None.
        """
        self.categoricalHelper      = categoricalHelper
        self.scorer                 = metrics.make_scorer(scoringFunction)
        self.searchType             = searchType
        self.searcher               = None

        # For testing various parameters.
        self.lastParameters         = ""
        self.lastScores             = ""
        self.testing                = testing


    def FitPredict(self, parameters, **kwargs):
        """
        Fits a hyperparameter search and runs predict.

        Parameters
        ----------
        parameters : dictionary
            Search parameters.
        **kwargs : keyword arguments
            These arguments are passed on to the searcher.

        Returns
        -------
        None.
        """
        self.Fit(parameters, **kwargs)
        self.Predict()


    def Fit(self, parameters, **kwargs):
        """
        Fits a cross validation search model.

        Parameters
        ----------
        parameters : dictionary
            Search parameters.
        **kwargs : keyword arguments
            These arguments are passed on to the searcher.

        Returns
        -------
        None.
        """
        # Make sure there is data to operate on.
        if len(self.categoricalHelper.dataHelper.xTrainingData) == 0:
            raise Exception("The data has not been split.")

        # Run the grid search.
        if self.searchType == "grid":
            self.searcher = GridSearchCV(self.categoricalHelper.model, parameters, scoring=self.scorer, **kwargs)
        elif self.searchType == "random":
            self.searcher = RandomizedSearchCV(self.categoricalHelper.model, parameters, scoring=self.scorer, **kwargs)

        self.searcher = self.searcher.fit(self.categoricalHelper.dataHelper.xTrainingData, self.categoricalHelper.dataHelper.yTrainingData)

        # Set the model (categoricalHelper) to the best combination of parameters.
        self.categoricalHelper.model = self.searcher.best_estimator_

        # Fit the best algorithm to the data.
        self.categoricalHelper.Fit()


    def Predict(self):
        self.categoricalHelper.Predict()


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


    def RunHypertuning(self, parameters, saveModel=True, **kwargs):
        """
        Provides a standard run sequence.

        Parameters
        ----------
        parameters : dictionary
            Parameters for the hypertuning.
        saveModel : bool, optional
            If true, the model is saved for later comparison to other models.
        **kwargs : keyword arguments
            These arguments are passed on to the hyperparameter search model at creation.

        Returns
        -------
        None.
        """
        # Store to use for comparison.
        if saveModel:
            ModelHelper.SaveModelHelper(self.categoricalHelper)

        # Set up the hyperparameter helper and run the grid search.
        self.FitPredict(parameters, **kwargs)

        if self.testing:
            self.categoricalHelper.dataHelper.consoleHelper.PrintNewLine(ConsoleHelper.VERBOSETESTING)
            self.categoricalHelper.dataHelper.consoleHelper.PrintBold("Last Parameters", ConsoleHelper.VERBOSETESTING)
            self.categoricalHelper.dataHelper.consoleHelper.Print(self.lastParameters, ConsoleHelper.VERBOSETESTING)
            self.categoricalHelper.dataHelper.consoleHelper.PrintNewLine(ConsoleHelper.VERBOSETESTING)
            self.categoricalHelper.dataHelper.consoleHelper.PrintBold("Last Scores", ConsoleHelper.VERBOSETESTING)
            self.categoricalHelper.dataHelper.consoleHelper.Print(self.lastScores, ConsoleHelper.VERBOSETESTING)

        self.DisplayChosenParameters()

        self.categoricalHelper.CreateConfusionMatrixPlot(dataSet="testing")
        self.categoricalHelper.DisplayModelPerformanceScores()

        self.lastParameters = self.searcher.best_params_
        self.lastScores     = self.categoricalHelper.GetModelPerformanceScores()