# -*- coding: utf-8 -*-
"""
Created on Mon Dec 27 19:30:11 2021

@author: Lance
"""
from IPython.display import display

import DataSetLoading
from lendres.BaggingHelper import BaggingHelper

import unittest

class TestBaggingHelper(unittest.TestCase):

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
        self.dataHelper         = TestBaggingHelper.dataHelper.Copy(deep=True)
        self.regressionHelper   = BaggingHelper(self.dataHelper)

        self.regressionHelper.SplitData(TestBaggingHelper.dependentVariable, 0.3)


    def testResults(self):
        self.regressionHelper.CreateModel()
        self.regressionHelper.Predict()
        self.regressionHelper.CreateConfusionMatrixPlot(dataSet="testing")
        result = self.regressionHelper.GetConfusionMatrix(dataSet="testing")
        display(result)
        result = self.regressionHelper.GetModelPerformanceScores()
        display(result)


if __name__ == "__main__":
    unittest.main()