# -*- coding: utf-8 -*-
"""
Created on Wed Jan 19 07:49:25 2022

@author: Lance
"""
from xgboost import XGBClassifier

from lendres.CategoricalRegressionHelper import CategoricalRegressionHelper

class XGradientBoostingHelper(CategoricalRegressionHelper):

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


    def CreateModel(self, **kwargs):
        """
        Creates a decision tree model.

        Parameters
        ----------
        **kwargs : keyword arguments
            These arguments are passed on to the XGBClassifier.

        Returns
        -------
        None.
        """

        if len(self.xTrainingData) == 0:
            raise Exception("The data has not been split.")

        self.model = XGBClassifier(eval_metric="logloss", use_label_encoder=False, random_state=1, **kwargs)
        self.model.fit(self.xTrainingData, self.yTrainingData)