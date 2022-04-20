"""
Created on January 19, 2022
@author: Lance
"""
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier

from lendres.CategoricalRegressionHelper import CategoricalRegressionHelper

class GradientBoostingHelper(CategoricalRegressionHelper):

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
        super().__init__(dataHelper, description)


    def CreateStandardModel(self, **kwargs):
        """
        Creates a decision tree model.

        Parameters
        ----------
        **kwargs : keyword arguments
            These arguments are passed on to the DecisionTreeClassifier.

        Returns
        -------
        None.
        """
        self.CreateModel(init=AdaBoostClassifier(random_state=1), **kwargs)


    def CreateModel(self, **kwargs):
        """
        Creates a decision tree model.

        Parameters
        ----------
        **kwargs : keyword arguments
            These arguments are passed on to the DecisionTreeClassifier.

        Returns
        -------
        None.
        """

        if len(self.dataHelper.xTrainingData) == 0:
            raise Exception("The data has not been split.")

        self.model = GradientBoostingClassifier(random_state=1, **kwargs)
        self.model.fit(self.dataHelper.xTrainingData, self.dataHelper.yTrainingData)