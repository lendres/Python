"""
Created on December 27, 2021
@author: Lance A. Endres
"""
import DataSetLoading
from lendres.DecisionTreeHelper                  import DecisionTreeHelper

import unittest

class TestDecisionTreeHelper(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.whichData = 1

        if cls.whichData == 0:
            cls.dataHelper, cls.dependentVariable = DataSetLoading.GetLoan_ModellingData()
        elif cls.whichData == 1:
            cls.dataHelper, cls.dependentVariable = DataSetLoading.GetCreditData()


    def setUp(self):
        """
        Set up function that runs before each test.  Creates a new copy of the data and uses
        it to create a new regression helper.
        """
        self.dataHelper         = TestDecisionTreeHelper.dataHelper.Copy(deep=True)
        self.dataHelper.SplitData(TestDecisionTreeHelper.dependentVariable, 0.3, stratify=True)

        self.regressionHelper   = DecisionTreeHelper(self.dataHelper, DecisionTreeHelper.CreateDefaultModel())


    def testStandardPlots(self):
        self.regressionHelper.FitPredict()
        self.regressionHelper.CreateDecisionTreePlot()
        self.regressionHelper.CreateFeatureImportancePlot()


    def testGetDependentVariableName(self):
        result = self.regressionHelper.dataHelper.GetDependentVariableName()
        self.assertEqual(result, TestDecisionTreeHelper.dependentVariable)


if __name__ == "__main__":
    unittest.main()