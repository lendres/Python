# -*- coding: utf-8 -*-
"""
Created on Mon Dec 27 19:30:11 2021

@author: Lance
"""
import DataSetLoading
from lendres.AdaBoostHelper import AdaBoostHelper

import unittest

class TestAdaBostHelper(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.dataHelper, cls.dependentVariable = DataSetLoading.GetCreditData(dropFirst=False)


    def setUp(self):
        """
        Set up function that runs before each test.  Creates a new copy of the data and uses
        it to create a new regression helper.
        """
        self.dataHelper         = TestAdaBostHelper.dataHelper.Copy(deep=True)
        self.regressionHelper   = AdaBoostHelper(self.dataHelper)

        self.regressionHelper.SplitData(TestAdaBostHelper.dependentVariable, 0.3, stratify=True)


    def testResults(self):
        self.regressionHelper.CreateModel()
        self.regressionHelper.Predict()
        self.regressionHelper.CreateConfusionMatrixPlot(dataSet="testing")

        result = self.regressionHelper.GetConfusionMatrix(dataSet="testing")
        self.assertAlmostEqual(result[1, 1], 43)

        result = self.regressionHelper.GetModelPerformanceScores()
        self.assertAlmostEqual(result.loc["Training", "Recall"], 0.5571, places=3)


if __name__ == "__main__":
    unittest.main()