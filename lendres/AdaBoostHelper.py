# -*- coding: utf-8 -*-
"""
Created on Wed Jan 19 07:49:25 2022

@author: Lance
"""
from sklearn.ensemble import AdaBoostClassifier

from lendres.CategoricalRegressionHelper import CategoricalRegressionHelper

class AdaBoostHelper(CategoricalRegressionHelper):

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

        if len(self.xTrainingData) == 0:
            raise Exception("The data has not been split.")

        self.model = AdaBoostClassifier(random_state=1, **kwargs)
        self.model.fit(self.xTrainingData, self.yTrainingData)