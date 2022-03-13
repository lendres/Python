# -*- coding: utf-8 -*-
"""
Created on Mon Dec 27 19:30:11 2021

@author: Lance
"""
from sklearn.linear_model import LogisticRegression

import DataSetLoading
from lendres.BaggingHelper import BaggingHelper

import unittest

class TestBaggingHelper(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.dataHelper, cls.dependentVariable = DataSetLoading.GetCreditData(dropFirst=False)


    def setUp(self):
        """
        Set up function that runs before each test.  Creates a new copy of the data and uses
        it to create a new regression helper.
        """
        self.dataHelper         = TestBaggingHelper.dataHelper.Copy(deep=True)
        self.regressionHelper   = BaggingHelper(self.dataHelper)

        self.regressionHelper.SplitData(TestBaggingHelper.dependentVariable, 0.3, stratify=True)


    def testDefaultResults(self):
        self.regressionHelper.CreateModel()
        self.regressionHelper.Predict()
        self.regressionHelper.CreateConfusionMatrixPlot(dataSet="testing")

        result = self.regressionHelper.GetConfusionMatrix(dataSet="testing")
        self.assertAlmostEqual(result[1, 1], 43)

        result = self.regressionHelper.GetModelPerformanceScores()
        self.assertAlmostEqual(result.loc["Training", "Recall"], 0.9428, places=3)


    def testLogisticRegressionClassifier(self):
        baseEstimator = LogisticRegression(solver='liblinear', max_iter=1000, random_state=1)
        self.regressionHelper.CreateModel(base_estimator=baseEstimator)
        self.regressionHelper.Predict()
        self.regressionHelper.CreateConfusionMatrixPlot(dataSet="testing")

        result = self.regressionHelper.GetConfusionMatrix(dataSet="testing")
        self.assertAlmostEqual(result[1, 1], 32)

        result = self.regressionHelper.GetModelPerformanceScores()
        self.assertAlmostEqual(result.loc["Training", "Recall"], 0.3380, places=3)

if __name__ == "__main__":
    unittest.main()