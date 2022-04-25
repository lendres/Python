"""
Created on January 19, 2022
@author: Lance
"""
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier

from lendres.CategoricalRegressionHelper import CategoricalRegressionHelper

class GradientBoostingHelper(CategoricalRegressionHelper):

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
        if model == None:
            model = GradientBoostingHelper.CreateDefaultModel()

        super().__init__(dataHelper, model, description)


    @classmethod
    def CreateStandardModel(self, **kwargs):
        """
        Creates a decision tree model.

        Parameters
        ----------
        **kwargs : keyword arguments
            These arguments are passed on to the DecisionTreeClassifier.

        Returns
        -------
        GradientBoostingClassifier.
        """
        return GradientBoostingHelper.CreateModel(init=AdaBoostClassifier(random_state=1), **kwargs)


    @classmethod
    def CreateDefaultModel(self, **kwargs):
        """
        Creates a decision tree model.

        Parameters
        ----------
        **kwargs : keyword arguments
            These arguments are passed on to the DecisionTreeClassifier.

        Returns
        -------
        GradientBoostingClassifier.
        """
        return GradientBoostingClassifier(random_state=1, **kwargs)