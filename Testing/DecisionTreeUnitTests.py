# -*- coding: utf-8 -*-
"""
Created on Mon Dec 27 19:30:11 2021

@author: Lance
"""
#from IPython.display import display

import DataSetLoading
from lendres.DecisionTreeHelper import DecisionTreeHelper

import unittest

class TestDecisionTreeHelper(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.whichData = 1

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
        self.dataHelper         = TestDecisionTreeHelper.dataHelper.Copy(deep=True)
        self.regressionHelper   = DecisionTreeHelper(self.dataHelper)

        self.regressionHelper.SplitData(TestDecisionTreeHelper.dependentVariable, 0.3, stratify=True)


    def testStandardPlots(self):
        self.regressionHelper.CreateModel()
        self.regressionHelper.CreateDecisionTreePlot()
        self.regressionHelper.CreateFeatureImportancePlot()


    def testGetDependentVariableName(self):
        result = self.regressionHelper.GetDependentVariableName()
        self.assertEqual(result, TestDecisionTreeHelper.dependentVariable)


if __name__ == "__main__":
    unittest.main()