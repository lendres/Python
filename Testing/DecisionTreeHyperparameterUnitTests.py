# -*- coding: utf-8 -*-
"""
Created on Mon Dec 27 19:30:11 2021

@author: Lance
"""
import numpy as np
#from IPython.display import display
from sklearn.metrics import recall_score

import DataSetLoading
from lendres.DecisionTreeHyperparameterHelper import DecisionTreeHyperparameterHelper

import unittest

class TestDecisionTreeHyperparameterHelper(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.whichData = 0

        if cls.whichData == 0:
            cls.dataHelper, cls.dependentVariable = DataSetLoading.GetLoan_ModellingData()
        elif cls.whichData == 1:
            cls.dataHelper, cls.dependentVariable = DataSetLoading.GetCreditData()

        #print("\nData size after cleaning:")
        #display(cls.dataHelper.data.shape)


    def setUp(self):
        """
        Set up function that runs before each test.  Creates a new copy of the data and uses
        it to create a new regression helper.
        """
        self.dataHelper         = TestDecisionTreeHyperparameterHelper.dataHelper.Copy(deep=True)
        self.regressionHelper   = DecisionTreeHyperparameterHelper(self.dataHelper)

        self.regressionHelper.SplitData(TestDecisionTreeHyperparameterHelper.dependentVariable, 0.3)


    def testHyperparameterTuning(self):
        parameters = {"max_depth"             : np.arange(1, 3),
                      "min_samples_leaf"      : [2, 5, 7],
                      "max_leaf_nodes"        : [2, 5, 10],
                      "criterion"             : ["entropy", "gini"]}

        self.regressionHelper.CreateModel()
        self.regressionHelper.CreateGridSearchModel(parameters, scoringFunction=recall_score)
        self.regressionHelper.DisplayChosenParameters()


if __name__ == "__main__":
    unittest.main()