# -*- coding: utf-8 -*-
"""
Created on Mon Dec 27 19:30:11 2021

@author: Lance
"""
import DataSetLoading

from lendres.RandomForestHelper import RandomForestHelper

import unittest

class TestRandomForestHelper(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.dataHelper, cls.dependentVariable = DataSetLoading.GetCreditData(dropFirst=False)


    def setUp(self):
        """
        Set up function that runs before each test.  Creates a new copy of the data and uses
        it to create a new regression helper.
        """
        self.dataHelper         = TestRandomForestHelper.dataHelper.Copy(deep=True)
        self.regressionHelper   = RandomForestHelper(self.dataHelper)

        self.regressionHelper.SplitData(TestRandomForestHelper.dependentVariable, 0.3, stratify=True)


    def testResults(self):
        print("\nTest 1")
        self.regressionHelper.CreateModel()
        self.regressionHelper.Predict()
        self.regressionHelper.CreateConfusionMatrixPlot(dataSet="testing")

        result = self.regressionHelper.GetConfusionMatrix(dataSet="testing")
        self.assertAlmostEqual(result[1, 1], 38)

        result = self.regressionHelper.GetModelPerformanceScores()
        self.assertAlmostEqual(result.loc["Testing", "Recall"], 0.4222, places=3)


if __name__ == "__main__":
    unittest.main()