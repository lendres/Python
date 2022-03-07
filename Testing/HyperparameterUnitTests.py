# -*- coding: utf-8 -*-
"""
Created on Mon Dec 27 19:30:11 2021

@author: Lance
"""
import numpy as np
#from IPython.display import display
from sklearn import metrics

import DataSetLoading
from lendres.DecisionTreeHelper import DecisionTreeHelper
from lendres.BaggingHelper import BaggingHelper
from lendres.RandomForestHelper import RandomForestHelper
from lendres.HyperparameterHelper import HyperparameterHelper

import unittest

class TestHyperparameterHelper(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.dataHelper, cls.dependentVariable = DataSetLoading.GetCreditData(dropFirst=False)

        #print("\nData size after cleaning:")
        #display(cls.dataHelper.data.shape)


    def setUp(self):
        """
        Set up function that runs before each test.  Creates a new copy of the data and uses
        it to create a new regression helper.
        """
        self.dataHelper             = TestHyperparameterHelper.dataHelper.Copy(deep=True)


    def testDecisiionTreeClassifier(self):
        parameters = {"max_depth"             : np.arange(1, 3),
                      "min_samples_leaf"      : [2, 5, 7],
                      "max_leaf_nodes"        : [2, 5, 10],
                      "criterion"             : ["entropy", "gini"]}

        self.regressionHelper       = DecisionTreeHelper(self.dataHelper)
        self.regressionHelper.SplitData(TestHyperparameterHelper.dependentVariable, 0.3)
        self.hyperparameterHelper   = HyperparameterHelper(self.regressionHelper)

        self.regressionHelper.CreateModel()
        self.hyperparameterHelper.CreateGridSearchModel(parameters, metrics.recall_score)
        self.hyperparameterHelper.DisplayChosenParameters()

        self.regressionHelper.Predict()
        self.regressionHelper.CreateConfusionMatrixPlot(dataSet="testing")


    def testBaggingClassifier(self):
        parameters = {"max_samples"  : [0.7, 0.8, 0.9, 1],
                      "max_features" : [0.7, 0.8, 0.9, 1],
                      "n_estimators" : [10,  20, 30, 40, 50]}

        self.regressionHelper       = BaggingHelper(self.dataHelper)
        self.regressionHelper.SplitData(TestHyperparameterHelper.dependentVariable, 0.3)
        self.hyperparameterHelper   = HyperparameterHelper(self.regressionHelper)

        self.regressionHelper.CreateModel()
        self.hyperparameterHelper.CreateGridSearchModel(parameters, metrics.recall_score)

        self.regressionHelper.Predict()
        self.regressionHelper.CreateConfusionMatrixPlot(dataSet="testing")


    def testRandomForestClassifier(self):
        parameters = {"n_estimators"     : [150, 200, 250],
                      "min_samples_leaf" : np.arange(5, 10),
                      "max_features"     : np.arange(0.2, 0.7, 0.1),
                      "max_samples"      : np.arange(0.3, 0.7, 0.1)}

        self.regressionHelper       = RandomForestHelper(self.dataHelper)
        self.regressionHelper.SplitData(TestHyperparameterHelper.dependentVariable, 0.3)
        self.hyperparameterHelper   = HyperparameterHelper(self.regressionHelper)

        self.regressionHelper.CreateModel()
        self.hyperparameterHelper.CreateGridSearchModel(parameters, metrics.recall_score)

        self.regressionHelper.Predict()
        self.regressionHelper.CreateConfusionMatrixPlot(dataSet="testing")


if __name__ == "__main__":
    unittest.main()