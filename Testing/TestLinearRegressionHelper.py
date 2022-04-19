# -*- coding: utf-8 -*-
"""
Created on Wed Jan 26 15:53:03 2022

@author: Lance
"""
#from IPython.display import display

import DataSetLoading
from lendres.DataHelper import DataHelper
from lendres.LinearRegressionHelper import LinearRegressionHelper

import unittest


class TestregressionHelper(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.dataHelper, cls.dependentVariable = DataSetLoading.GetInsuranceData()


    def setUp(self):
        """
        Set up function that runs before each test.  Creates a new copy of the data and uses
        it to create a new regression helper.
        """
        dataHelper = DataHelper.Copy(TestregressionHelper.dataHelper, deep=True)
        self.regressionHelper = LinearRegressionHelper(dataHelper)
        self.regressionHelper.SplitData("charges", 0.3, stratify=False)
        self.regressionHelper.CreateModel()


    def testModelCoefficients(self):
        result = self.regressionHelper.GetModelCoefficients()
        #print(result)
        self.assertAlmostEqual(result["Coefficients"]["age"], 251.681865, places=3)


    def testPerformanceScores(self):
        self.regressionHelper.Predict()
        result = self.regressionHelper.GetModelPerformanceScores()
        self.assertAlmostEqual(result.loc["Testing", "RMSE"], 6063.122657, places=3)


if __name__ == "__main__":
    unittest.main()