# -*- coding: utf-8 -*-
"""
Created on Mon Dec 27 19:30:11 2021

@author: Lance
"""
from IPython.display import display

import DataSetLoading
#from lendres.ConsoleHelper import ConsoleHelper
#from lendres.DataHelper import DataHelper
from lendres.RandomForestHelper import RandomForestHelper

import unittest

class TestRandomForestHelper(unittest.TestCase):

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
        self.dataHelper         = TestRandomForestHelper.dataHelper.Copy(deep=True)
        self.regressionHelper   = RandomForestHelper(self.dataHelper)

        self.regressionHelper.SplitData(TestRandomForestHelper.dependentVariable, 0.3)


    def testResults(self):
        print("\nTest 1")
        self.regressionHelper.CreateModel()
        self.regressionHelper.Predict()
        self.regressionHelper.CreateConfusionMatrixPlot(dataSet="testing")
        result = self.regressionHelper.GetConfusionMatrix(dataSet="testing")
        display(result)
        result = self.regressionHelper.GetModelPerformanceScores()
        display(result)


if __name__ == "__main__":
    unittest.main()