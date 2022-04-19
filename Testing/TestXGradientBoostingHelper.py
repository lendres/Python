# -*- coding: utf-8 -*-
"""
Created on Mon Dec 27 19:30:11 2021

@author: Lance
"""
from IPython.display import display

import DataSetLoading
from lendres.ConsoleHelper           import ConsoleHelper
from lendres.XGradientBoostingHelper import XGradientBoostingHelper

import unittest

class TestXGradientBoostingHelper(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        cls.dataHelper, cls.dependentVariable = DataSetLoading.GetCreditData(verboseLevel=ConsoleHelper.VERBOSEREQUESTED, dropFirst=False)

        #print("\nData size after cleaning:")
        #display(cls.dataHelper.data.shape)


    def setUp(self):
        """
        Set up function that runs before each test.  Creates a new copy of the data and uses
        it to create a new regression helper.
        """
        self.dataHelper         = TestXGradientBoostingHelper.dataHelper.Copy(deep=True)
        self.regressionHelper   = XGradientBoostingHelper(self.dataHelper)

        self.regressionHelper.SplitData(TestXGradientBoostingHelper.dependentVariable, 0.3, stratify=True)


    def testResults(self):
        self.regressionHelper.CreateModel()
        self.regressionHelper.Predict()
        self.regressionHelper.CreateConfusionMatrixPlot(dataSet="testing")
        result = self.regressionHelper.GetConfusionMatrix(dataSet="testing")
        self.assertAlmostEqual(result[1, 1], 48)
        result = self.regressionHelper.GetModelPerformanceScores()
        #display(result)
        self.assertAlmostEqual(result.loc["Testing", "Recall"], 0.5333, places=3)


if __name__ == "__main__":
    unittest.main()