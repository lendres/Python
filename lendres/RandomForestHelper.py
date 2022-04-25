"""
Created on January 19, 2022
@author: Lance
"""
from sklearn.ensemble import RandomForestClassifier

from lendres.CategoricalRegressionHelper import CategoricalRegressionHelper

class RandomForestHelper(CategoricalRegressionHelper):

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
            model = RandomForestHelper.CreateDefaultModel()

        super().__init__(dataHelper, model, description)


    @classmethod
    def CreateDefaultModel(cls, **kwargs):
        """
        Creates a decision tree model.

        Parameters
        ----------
        **kwargs : keyword arguments
            These arguments are passed on to the DecisionTreeClassifier.

        Returns
        -------
        RandomForestClassifier.
        """
        return RandomForestClassifier(random_state=1, **kwargs)