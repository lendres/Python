# -*- coding: utf-8 -*-
"""
Created on Wed Jan 26 15:53:03 2022

@author: Lance
"""
#from IPython.display import display
import os

from lendres.ConsoleHelper import ConsoleHelper
from lendres.DataHelper import DataHelper
from lendres.LinearRegressionHelper import LinearRegressionHelper
import unittest


class TestLinearRegressionHelper(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        inputFile       = "insurance.csv"

        inputFile       = os.path.join("Data", inputFile)
        print(inputFile)

        consoleHelper   = ConsoleHelper(verboseLevel=ConsoleHelper.VERBOSENONE)
        cls.loanData    = DataHelper(consoleHelper=consoleHelper)

        cls.loanData.LoadAndInspectData(inputFile)
        cls.loanData.EncodeCategoricalColumns(["region", "sex", "smoker"])


    def setUp(self):
        """
        Set up function that runs before each test.  Creates a new copy of the data and uses
        it to create a new regression helper.
        """
        loanData = DataHelper.Copy(TestLinearRegressionHelper.loanData, deep=True)
        self.linearRegressionHelper = LinearRegressionHelper(loanData)
        self.linearRegressionHelper.SplitData("charges", 0.3)
        self.linearRegressionHelper.CreateModel()


    def testModelCoefficients(self):
        result = self.linearRegressionHelper.GetModelCoefficients()
        #print(result)
        self.assertAlmostEqual(result["Coefficients"]["age"], 251.681865, places=3)



    def testPerformanceScores(self):
        self.linearRegressionHelper.Predict()
        result = self.linearRegressionHelper.GetModelPerformanceScores()
        self.assertAlmostEqual(result.loc["Testing", "RMSE"], 6063.122657, places=3)


if __name__ == "__main__":
    unittest.main()