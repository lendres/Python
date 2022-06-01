"""
Created on January 19, 2022
@author: Lance A. Endres
"""
from sklearn.ensemble                    import BaggingClassifier

from lendres.CategoricalRegressionHelper import CategoricalRegressionHelper

class BaggingHelper(CategoricalRegressionHelper):

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
            model = BaggingHelper.CreateDefaultModel()

        super().__init__(dataHelper, model, description)


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
        BaggingClassifier.
        """
        return BaggingClassifier(random_state=1, **kwargs)