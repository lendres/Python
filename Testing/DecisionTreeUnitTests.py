# -*- coding: utf-8 -*-
"""
Created on Mon Dec 27 19:30:11 2021

@author: Lance
"""
import pandas as pd
import numpy as np
from IPython.display import display
from sklearn.metrics import recall_score

import lendres
from lendres.DecisionTreeHelper import DecisionTreeHelper
import unittest

class TestDecisionTreeHelper(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        inputFile = "backpain.csv"
        #cls.data = lendres.Data.LoadAndInspectData(inputFile)
        cls.data   = pd.read_csv(inputFile)


    def setUp(self):
        """
        Set up function that runs before each test.  Creates a new copy of the data and uses
        it to create a new regression helper.
        """
        self.data             = TestDecisionTreeHelper.data.copy(deep=True)
        self.regressionHelper = DecisionTreeHelper(self.data)

        columnAsNumeric       = self.regressionHelper.ConvertCategoryToNumeric("Status", "Abnormal")
        self.regressionHelper.SplitData(columnAsNumeric, 0.3)
   
        
    def testStandardPlots(self):
        self.regressionHelper.CreateModel()
        self.regressionHelper.CreateDecisionTreePlot()
        self.regressionHelper.CreateFeatureImportancePlot()
        self.regressionHelper.CreateConfusionMatrixPlot(dataSet="training")
        self.regressionHelper.CreateConfusionMatrixPlot(dataSet="testing")


    def testGetDependentVariableName(self):
        result = self.regressionHelper.GetDependentVariableName()
        self.assertEqual(result, "Status_int")


    def testHyperparameterTuning(self):
        parameters = {"max_depth"             : np.arange(1, 3),
                      "min_samples_leaf"      : [2, 5, 7],
                      "max_leaf_nodes"        : [2, 5, 10],
                      "criterion"             : ["entropy", "gini"]
                     }

        self.regressionHelper.CreateModel()
        self.regressionHelper.CreateGridSearchModel(parameters, scoringFunction=recall_score)


    def testCostComplexityPruningModel(self):
        self.regressionHelper.CreateModel()
        self.regressionHelper.CreateCostComplexityPruningModel("recall")
        self.regressionHelper.Predict()
        result = self.regressionHelper.GetModelPerformanceScores()
        self.assertAlmostEqual(result.loc["Training", "Accuracy"], 0.838710, places=6)
        self.assertAlmostEqual(result.loc["Testing", "Recall"], 0.863636, places=6)


    def testCostComplexityPrusingPlot(self):
        self.regressionHelper.CreateModel()
        self.regressionHelper.CreateCostComplexityPruningModel("accuracy")
        self.regressionHelper.CreateAlphasVersusScoresPlot("accuracy")


if __name__ == "__main__":
    unittest.main()